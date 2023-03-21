"""
    Inference script for sbi.

    This script performs inference on a test dataset in a simulation-based inference framework with the SBI package.
    It loads a density estimator trained to approximate the posterior distribution for a dataset of simulated data
    and checks its performance on a test dataset.
    Simulation-based calibration is also performed to check if the posterior is well behaving.
    See https://www.mackelab.org/sbi/ for more details.

     Running the code:

        python3 infer_test_sbi.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import collections
import pickle

import numpy as np
import torch
from sbi import utils
from sbi.analysis import check_sbc, run_sbc, sbc_rank_plot
from sbi.inference import SNPE

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device


def infer(args, config):

    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("Inference")
    logger.info("Logger initialized...")

    # Show experiment information ------------------------------------------
    logger.info("=========================================================")

    dataset_path = config["test_data_loader"]["dataset_path"]
    dataset_stat_path = config["test_data_loader"]["statistic_path"]
    filter_inputs = config["test_data_loader"]["filter_inputs"]
    filter_labels = config["test_data_loader"]["filter_labels"]
    normalize = config["test_data_loader"]["normalize"]
    standardize = config["test_data_loader"]["standardize"]
    input_shape = config["arch"]["args"]["input_shape"]
    hidden_features = config["arch"]["args"]["len_output_layer"]
    n_parameters = len(filter_labels)

    # Set up GPU device if available.
    logger.info("Requesting {} GPUs...".format(config["n_gpu"]))
    device, device_ids = request_device(logger, config["n_gpu"])
    logger.info("Devices obtained: {}".format(device_ids))

    # Load the test dataset ----------------------------------------------------------
    logger.info("Loading the test dataset...")
    dataset = dl.DatasetMultichannelArray(
        dataset_path=dataset_path,
        statistic_path=dataset_stat_path,
        filter_channels=filter_inputs,
        filter_labels=filter_labels,
        normalize=normalize,
        standardize=standardize,
    )

    parameter = np.zeros((len(dataset), n_parameters))
    matrix = np.zeros(
        (len(dataset), 1, input_shape[0], input_shape[1], input_shape[2])
    )
    for i, (x, theta) in enumerate(dataset):
        # Re-shape the matrix to have the channel number at the beginning and add an extra
        # dimension that is needed for sbi.
        x = np.moveaxis(x, -1, 0)
        matrix[i] = x[None, :]
        parameter[i] = theta

    # Transform the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)

    # Build embedding model ------------------------------------------------
    # The weights are initialized with this procedure only for the embedding net.
    logger.info("Building embedding model...")
    embedding_net = config.init_object("arch", learning_models)
    logger.info("Model architecture: {}".format(embedding_net))

    # Build density estimator ----------------------------------------------
    # The default mixture density estimator has 3 hidden layers with a number of neurons = hidden_features.
    # The weights are initialized with the default initialization provided by pytorch.
    neural_posterior = utils.posterior_nn(
        model=config["density_estimator"]["type"],
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_components=config["density_estimator"]["args"]["num_components"],
        device=device,
    )

    # Set prior distribution for the parameters ------------------------------------------
    logger.info("Set prior distribution...")
    if normalize:
        # All the parameters are rescaled in the range [0, 1].
        prior = utils.BoxUniform(
            low=torch.tensor(np.zeros(n_parameters)),
            high=torch.tensor(np.ones(n_parameters)),
            device=f"{device}",
        )
    elif standardize:
        # All the parameters are rescaled so that they have mean 0 and std 1.
        # We consider a range of 5 std [-5, 5].
        prior = utils.BoxUniform(
            low=torch.tensor(-5.0 * np.ones(n_parameters)),
            high=torch.tensor(5.0 * np.ones(n_parameters)),
            device=f"{device}",
        )
    else:
        prior = utils.BoxUniform(
            low=torch.tensor(dataset.target_min),
            high=torch.tensor(dataset.target_max),
            device=f"{device}",
        )

    # Set up the inference procedure -----------------------------
    # By default the procedure is the SNPE-C (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
    inference = SNPE(
        prior=prior, density_estimator=neural_posterior, device=f"{device}"
    )

    # Load the trained model.
    logger.info("Loading the trained model...")
    with open(args.weights, "rb") as f:
        trained_model = pickle.load(f)

    # Build the posterior.
    posterior = inference.build_posterior(trained_model.to(device))

    # Compute the average loss over the test dataset (with batch size = 1).
    logger.info("Computing the average loss over the test dataset...")
    test_loss_mean = torch.tensor([0.0]).to(device)
    for i in range(len(dataset)):
        test_loss_mean += posterior.log_prob(
            parameter[i].to(device), matrix[i].to(device)
        )

    test_loss_mean = test_loss_mean / len(dataset)
    logger.info("Average loss of the test dataset: {}".format(test_loss_mean))

    logger.info("Perform Simulation Based Calibration...")
    # Run SBC: for each test sample we draw 1000 posterior samples.
    num_posterior_samples = 1000
    ranks, dap_samples = run_sbc(
        parameter.to(device),
        matrix.to(device),
        posterior,
        num_posterior_samples=num_posterior_samples,
    )

    logger.info("Check the rank statistics...")
    # Check if the rank distributions follow a uniform distribution with three different tests
    # (see [here](https://www.mackelab.org/sbi/tutorial/13_diagnostics_simulation_based_calibration/)
    # for more details on these tests).
    check_stats = check_sbc(
        ranks,
        parameter.to(device),
        dap_samples.to(device),
        num_posterior_samples=num_posterior_samples,
    )
    logger.info(
        f"kolmogorov-smirnov p-values \n - check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"
    )
    logger.info(
        f"c2st accuracies \n - check_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()} "
        f"\n - check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}"
    )

    # Visually check if the ranks follow a uniform distribution.
    # The gray band represents the 99% credibility interval around the mean for a uniform distribution.
    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="hist",
        num_bins=30,  # By passing None the default is len(dataset_test) / 20.
    )

    f.savefig(f"{config.log_dir}/ranks_histograms.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="PyPopSyn simulation based inference"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="examples/learning/config_sbi.json",
        help="Configuration file path.",
    )

    args.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained model.",
    )

    args.add_argument(
        "--infer",
        nargs="?",
        type=str,
        default=True,
        help="Flag to set up the inference saving path. If False you are in training mode.",
    )

    CustomArgs = collections.namedtuple(
        "CustomArgs", "flags type nargs target"
    )

    options = [
        CustomArgs(
            ["--dataset"],
            type=str,
            nargs="?",
            target="test_data_loader;dataset_path",
        ),
        CustomArgs(
            ["--dataset_statistics"],
            type=str,
            nargs="?",
            target="test_data_loader;statistic_path",
        ),
        CustomArgs(
            ["--filter_inputs"],
            type=int,
            nargs="*",
            target="test_data_loader;filter_inputs",
        ),
        CustomArgs(
            ["--filter_labels"],
            type=int,
            nargs="*",
            target="test_data_loader;filter_labels",
        ),
        CustomArgs(
            ["--input_shape"],
            type=int,
            nargs=3,
            target="arch;args;input_shape",
        ),
        CustomArgs(
            ["--len_output_layer"],
            type=int,
            nargs="?",
            target="arch;args;len_output_layer",
        ),
        CustomArgs(
            ["--normalize"],
            type=bool,
            nargs="?",
            target="test_data_loader;normalize",
        ),
        CustomArgs(
            ["--standardize"],
            type=bool,
            nargs="?",
            target="test_data_loader;standardize",
        ),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        args,
        options,
    )

    infer(args.parse_args(), configuration)
