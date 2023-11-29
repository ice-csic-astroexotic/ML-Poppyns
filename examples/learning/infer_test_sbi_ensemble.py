"""
    Inference script for sbi.

    This script performs inference on a test dataset in a simulation-based inference framework with the SBI package.
    It loads a density estimator trained to approximate the posterior distribution for a dataset of simulated data
    and checks its performance on a test dataset.
    Simulation-Based Calibration is also performed to check if the posterior is well behaving.
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
import json
import os
import pathlib
import pickle
import sys
import time

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.analysis import check_sbc, run_sbc, sbc_rank_plot
from sbi.inference import SNPE
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device


def calculate_smallest_hdr(
    norm_exp: list,
    posterior_ensemble: callable,
    true_value: torch.tensor,
    posterior_samples_std: torch.tensor,
    posterior_samples_norm: torch.tensor,
    simulation_output: torch.tensor,
    device: str,
) -> float:

    """
    Calculating the smallest highest density region of the posterior, that contains the true value.

    Args:
        norm_exp (List): List of booleans indicating whether we apply normalization or standardization.
        posterior_ensemble (Callable): Ensemble posterior distribution function.
        true_value (torch.tensor): Tensor containing the values of the parameters used to generate the simulated population
         in simulation_output.
        posterior_samples_std (torch.tensor): Tensor containing the samples standrized from the inferred ensemble
        posterior distribution for simulation_output.
        posterior_samples_norm (torch.tensor): Tensor containing the samples normalized from the inferred ensemble
        posterior distribution for simulation_output.
        simulation_output (torch.tensor): Tensor containing the maps of the simulated population.
        device (str): String specifying the type of the device used to run the script.

    Returns:
        hdr (float): Smallest highest density region of the posterior that contains the true value.
    """
    log_p_samples = np.zeros(len(posterior_samples_norm))
    log_prob_true_value = 0

    for index, posterior in enumerate(posterior_ensemble):

        # Evaluating the PDF value of the ground truth.
        log_prob_true_value_exp = (
            posterior.log_prob(
                true_value[index].to(device),
                simulation_output[index].to(device),
            )
            .cpu()
            .numpy()[0]
        )
        log_prob_true_value += log_prob_true_value_exp

        # Evaluating the PDF values of the posterior samples. To do so, we need to choose 'std' or 'norm' depending on each experiment.
        if norm_exp:
            posterior_samples = posterior_samples_norm
        else:
            posterior_samples = posterior_samples_std

        log_p_samples_exp = [
            posterior.log_prob(
                posterior_samples[i].to(device),
                simulation_output[index].to(device),
            )
            .cpu()
            .numpy()[0]
            for i in range(len(posterior_samples))
        ]
        log_p_samples += log_p_samples_exp

    log_p_samples /= len(log_p_samples_exp)
    log_prob_true_value_exp /= len(posterior_ensemble)

    # Determining the fraction of PDF values that are larger than that of the ground truth.
    hdr = (log_p_samples > log_prob_true_value_exp).mean()
    return hdr


def infer(args, config):

    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("Inference")
    logger.info("Logger initialized...")

    # Initialize the path where the time profiling will be saved.
    prof_log_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_log"])
    )
    prof_json_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_json"])
    )

    # Remove the profile.json and profile.log files to prevent interrupted server connections issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)

    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

    # Show experiment information ------------------------------------------
    logger.info("=========================================================")

    with timewith.TimeWith(
        "[TotalInference]",
        prof_log_path,
        prof_json_path,
        config["show_profiling"],
    ):
        with timewith.TimeWith(
            "[Initialization]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):
            # Set up GPU device if available.
            logger.info("Requesting {} GPUs...".format(config["n_gpu"]))
            device, device_ids = request_device(logger, config["n_gpu"])
            logger.info("Devices obtained: {}".format(device_ids))

            posterior_ensemble = []

            # Opening the txt file with all the path to the trained models.
            with open(args.trained_model, "rb") as f:
                trained_models_path = f.readlines()

            dataset_exps = []
            parameter_exps = []
            matrix_exps = []
            norm_exp = []
            # We loop through all the different experiments.
            for index, exp in enumerate(trained_models_path):
                # Extracting the path of the trained model and the config associated to each experiment.
                model_path = exp.strip().split()[0]
                config_path = exp.strip().split()[1]
                with open(model_path, "rb") as f:
                    trained_model = pickle.load(f)
                with open(config_path, "rb") as f_config:
                    config_json = json.load(f_config)

                # Initialize the torch seed.
                if config["set_manual_seed"] is True:
                    torch.manual_seed(config["manual_seed"])
                    logger.info("Seed: {}".format(config["manual_seed"]))
                else:
                    torch.manual_seed(int(time.time()))
                    logger.info("Seed: {}".format(int(time.time())))

                dataset_path = config_json["test_data_loader"]["dataset_path"]
                dataset_stat_path = config_json["test_data_loader"][
                    "statistic_path"
                ]
                filter_inputs = config_json["test_data_loader"][
                    "filter_inputs"
                ]
                filter_labels = config_json["test_data_loader"][
                    "filter_labels"
                ]
                normalize = config_json["test_data_loader"]["normalize"]
                standardize = config_json["test_data_loader"]["standardize"]
                input_shape = config_json["arch"]["args"]["input_shape"]
                hidden_features = config_json["arch"]["args"][
                    "len_output_layer"
                ]
                n_components = config_json["density_estimator"]["args"][
                    "num_components"
                ]
                n_parameters = len(filter_labels)
                norm_exp.append(normalize)

                with timewith.TimeWith(
                    "[TestDatasetLoader]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):

                    # Load the test dataset ----------------------------------------------------------
                    logger.info("Loading the test dataset...")
                    try:
                        dataset = dl.DatasetMultichannelArray(
                            dataset_path=dataset_path,
                            statistic_path=dataset_stat_path,
                            filter_channels=filter_inputs,
                            filter_labels=filter_labels,
                            normalize=normalize,
                            standardize=standardize,
                        )
                    except Exception:
                        logger.exception("Error: an error occurred:")
                        sys.exit(1)

                    parameter = np.zeros((len(dataset), n_parameters))
                    matrix = np.zeros(
                        (
                            len(dataset),
                            1,
                            input_shape[0],
                            input_shape[1],
                            input_shape[2],
                        )
                    )
                    for i, (x, theta) in enumerate(dataset):
                        # Reshape the matrix to have the channel number at the beginning
                        x = np.moveaxis(x, -1, 0)

                        if list(x.shape) != input_shape:
                            logger.error(
                                "Mismatch between the shape of the input data x {} and the input shape specified "
                                "in the configuration file {}".format(
                                    x.shape, input_shape
                                )
                            )
                            sys.exit()

                        matrix[i] = x
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
                    # The weights are initialized with the default initialization provided by PyTorch.
                    neural_posterior = utils.posterior_nn(
                        model=config["density_estimator"]["type"],
                        embedding_net=embedding_net,
                        hidden_features=hidden_features,
                        num_components=n_components,
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
                        # Set the prior range to the range of the parameters.
                        prior = utils.BoxUniform(
                            low=torch.tensor(dataset.target_min),
                            high=torch.tensor(dataset.target_max),
                            device=f"{device}",
                        )

                    # Set up the inference procedure -----------------------------
                    # By default the procedure is the SNPE-C (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
                    inference = SNPE(
                        prior=prior,
                        density_estimator=neural_posterior,
                        device=f"{device}",
                    )

                    # Load the trained models from the txt file.
                    logger.info("Loading the trained models...")
                    logger.info(
                        "Inference is performed with the trained models in: {}".format(
                            args.trained_model
                        )
                    )

                    inference_model = inference.build_posterior(
                        trained_model.to(device)
                    )

                    dataset_exps.append(dataset)
                    parameter_exps.append(parameter)
                    matrix_exps.append(matrix)
                    posterior_ensemble.append(inference_model)

        with timewith.TimeWith(
            "[Inference]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):

            # Compute the loss over the test dataset (with batch size = 1), extract the Gaussian mixture
            # coefficients and generate the corner plots.
            logger.info(
                "Computing the average loss over the test dataset, extracting Gaussian mixture coefficients, estimating the hdr for the coverage probability and generating corner plots...."
            )
            test_loss_mean_ensemble = torch.tensor([0.0]).to(device)

            hdr_testset = np.zeros(len(dataset))

            for i in range(len(dataset)):

                test_loss_mean = torch.tensor([0.0]).to(device)
                n_samples_coverage_exp = 1000
                posterior_samples_coverage_norm = []
                posterior_samples_coverage_std = []
                parameter_sample = []
                matrix_sample = []

                for index, posterior in enumerate(posterior_ensemble):

                    parameter = parameter_exps[index]
                    matrix = matrix_exps[index]
                    test_loss_mean += posterior.log_prob(
                        parameter[i].to(device), matrix[i].to(device)
                    )
                    posterior_sample_exp = posterior.set_default_x(
                        matrix[i]
                    ).sample(
                        (n_samples_coverage_exp,), show_progress_bars=False
                    )

                    # Save the statistics for the filtered labels. Here we assume that the test datasets is the same for
                    # all the experiments.
                    par_max = torch.tensor(dataset.target_max)
                    par_min = torch.tensor(dataset.target_min)
                    par_std = torch.tensor(dataset.target_std)
                    par_mean = torch.tensor(dataset.target_mean)

                    # If the parameters were normalized or standardized rescale quantities to their physical ranges on
                    # order to concatenate all together for calculating the coverage.
                    if norm_exp[index]:
                        posterior_sample_exp_phys = (
                            posterior_sample_exp * (par_max - par_min)
                            + par_min
                        )
                        posterior_sample_exp_std = (
                            posterior_sample_exp_phys - par_mean
                        ) / par_std
                        posterior_samples_coverage_std.append(
                            posterior_sample_exp_std
                        )
                        posterior_samples_coverage_norm.append(
                            posterior_sample_exp
                        )
                    else:
                        posterior_sample_exp_phys = (
                            posterior_sample_exp * par_std + par_mean
                        )
                        posterior_sample_exp_norm = (
                            posterior_sample_exp_phys - par_min
                        ) / (par_max - par_min)
                        posterior_samples_coverage_std.append(
                            posterior_sample_exp
                        )
                        posterior_samples_coverage_norm.append(
                            posterior_sample_exp_norm
                        )

                    parameter_sample.append(parameter[i])
                    matrix_sample.append(matrix[i])

                posterior_samples_coverage_std = torch.cat(
                    posterior_samples_coverage_std
                )
                posterior_samples_coverage_norm = torch.cat(
                    posterior_samples_coverage_norm
                )

                test_loss_mean_ensemble += test_loss_mean / len(
                    posterior_ensemble
                )

                smallest_hdr = calculate_smallest_hdr(
                    norm_exp,
                    posterior_ensemble,
                    parameter_sample,
                    posterior_samples_coverage_std,
                    posterior_samples_coverage_norm,
                    matrix_sample,
                    device,
                )

                hdr_testset[i] = smallest_hdr
                # If the "corner plot" argument is set to True, we draw samples from the inferred posterior
                # distribution. Moreover, we save these samples and the corresponding corner plot.
                """
                if args.corner_plot:

                    samples = (
                        ensemble_posteriors.set_default_x(matrix[i])
                        .sample((50000,), show_progress_bars=False)
                        .cpu()
                    )

                    # Save the statistics for the filtered labels.
                    par_max = torch.tensor(dataset.target_max)
                    par_min = torch.tensor(dataset.target_min)
                    par_std = torch.tensor(dataset.target_std)
                    par_mean = torch.tensor(dataset.target_mean)

                    # If the parameters were normalized or standardized rescale quantities to their physical ranges.
                    if normalize:
                        parameter[i] = (
                            parameter[i] * (par_max - par_min) + par_min
                        )
                        samples = samples * (par_max - par_min) + par_min

                    elif standardize:
                        parameter[i] = parameter[i] * par_std + par_mean
                        samples = samples * par_std + par_mean

                    # Save the samples from the inferred posterior distribution.
                    torch.save(
                        samples,
                        f"{config.log_dir}/samples_{i}.pt",
                    )

                    # Saving the best estimated parameters and the 95% CI into the log.txt file.
                    quantile = np.quantile(
                        samples, [0.025, 0.5, 0.975], axis=0
                    )

                    logger.info(
                        "Estimated parameter values (we consider the median as the best value and the 95 % credibility interval):"
                    )

                    for s in range(n_parameters):

                        param_quantile = quantile[:, s]
                        logger.info(
                            f"{parameter_labels[s]} = {param_quantile[1]} + {param_quantile[2] - param_quantile[1]} - {param_quantile[1] - param_quantile[0]}"
                        )

                    range_param = [
                        [par_min[v], par_max[v]] for v in range(len(par_max))
                    ]

                    param_median = quantile[1, :]

                    # Corner plot of the inferred posterior distributions for each parameter.
                    figure = corner.corner(
                        samples.detach().cpu().numpy(),
                        bins=32,
                        labels=parameter_labels,
                        range=range_param,
                        quantiles=[0.025, 0.5, 0.975],
                        levels=(
                            1 - np.exp(-0.5),
                            1 - np.exp(-2),
                            1 - np.exp(-9.0 / 2.0),
                        ),  # 1, 2 and 3 sigma levels
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
                    corner.overplot_lines(
                        figure, param_median, color="tab:red"
                    )
                    corner.overplot_lines(
                        figure, parameter[i], color="tab:blue"
                    )
                    corner.overplot_points(
                        figure,
                        param_median[None],
                        marker="s",
                        color="tab:red",
                    )
                    corner.overplot_points(
                        figure,
                        parameter[None, i],
                        marker="s",
                        color="tab:blue",
                    )
                    plt.savefig(f"{config.log_dir}/corner_plot_{i}.pdf")
                    plt.close()
                    """

            logger.info("Computing the coverage probability...")
            # Calculate the coverage from the smallest hdr.
            betas = np.linspace(0, 1, 12)
            coverage_probability = []
            hdr_testset_sorted = np.sort(np.asarray(hdr_testset))

            # For each value of beta, we calculate the percentage of test samples for which the
            # highest density region (HDR), in hdr_testset, is lower than beta. Then, among these test samples, each of the
            # true values will fall inside this beta's HDR.

            for beta in betas:
                coverage_probability.append((hdr_testset_sorted < beta).mean())

            # Saving the coverage probability array to reproduce the coverage plot.
            np.save(
                f"{config.log_dir}/coverage_probability.npy",
                coverage_probability,
            )

            # Plot the coverage.
            plt.plot(
                betas,
                coverage_probability,
                color="steelblue",
                label="upper right",
            )
            plt.plot([0, 1], [0, 1], color="k", linestyle="--")
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.xlabel(r"Credibility level $1-\alpha$", fontsize=10)
            plt.ylabel(r"Coverage probability", fontsize=10)
            plt.savefig(f"{config.log_dir}/coverage_plot.pdf")

            # To save the precision and means for the whole test dataset in a numpy array uncomment the lines below.
            # np.save(f"{config.log_dir}/mean_Gaussians.npy",mean_Gaussians)
            # np.save(f"{config.log_dir}/precision_Gaussians.npy",precision_Gaussians)

            # Divide the cumulative test loss by the number of samples to obtain the average loss over the test set.
            test_loss_mean = test_loss_mean / len(dataset)
            logger.info(
                "Average loss over the test dataset: {}".format(test_loss_mean)
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="PyPopSyn Simulation-Based Inference"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="examples/learning/config_sbi.json",
        help="Configuration file path.",
    )
    args.add_argument(
        "--corner_plot",
        type=bool,
        default=False,
        help="If a posterior corner plot for each test sample is required, this argument should be set to True.",
    )

    args.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="Path to txt with the path of the trained models.",
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
