"""
    Training script for sbi.

    This script carries out the training in a simulation-based inference framework with the SBI package.
    It trains a density estimator to approximate the posterior distribution for a dataset of simulated data.
    Note that with this method we can evaluate the posterior for different simulated populations without
    having to re-train the model. This is called amortization. An amortized posterior is one that is not
    focused on any particular observation. See https://www.mackelab.org/sbi/ for more details.

    Running the code:

        python3 train_sbi.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import collections
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import utils
from sbi.analysis import tensorboard_output as tbo
from sbi.inference import SNPE

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device


def train(config):

    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("train")
    logger.info("Logger initialized...")

    # Show experiment information ------------------------------------------
    logger.info("=========================================================")

    # Initialize the torch seed.
    if config["set_manual_seed"] is True:
        torch.manual_seed(config["manual_seed"])
        logger.info("Seed: {}".format(config["manual_seed"]))
    else:
        logger.info("Seed: {}".format(torch.seed()))

    logger.info("Train configuration: {}".format(config._configuration))

    dataset_path = config["training_data_loader"]["dataset_path"]
    dataset_stat_path = config["training_data_loader"]["statistic_path"]
    filter_inputs = config["training_data_loader"]["filter_inputs"]
    filter_labels = config["training_data_loader"]["filter_labels"]
    normalize = config["training_data_loader"]["normalize"]
    standardize = config["training_data_loader"]["standardize"]
    input_shape = config["arch"]["args"]["input_shape"]
    hidden_features = config["arch"]["args"]["len_output_layer"]
    n_parameters = len(filter_labels)

    # Set up GPU device if available.
    logger.info("Requesting {} GPUs...".format(config["n_gpu"]))
    device, device_ids = request_device(logger, config["n_gpu"])
    logger.info("Devices obtained: {}".format(device_ids))

    # Build embedding model ------------------------------------------------
    logger.info("Building embedding model...")
    embedding_net = config.init_object("arch", learning_models)
    logger.info("Model architecture: {}".format(embedding_net))

    # Initialize weights ---------------------------------------------------
    # The weights are initialized with this procedure only for the embedding net.
    logger.info("Initializing weights...")
    weight_initializer = config.init_object(
        "weights_initializer", learning_initializers
    )
    logger.info("Weight initialization: {}".format(weight_initializer))
    # Apply the weight initialization scheme to every layer in the model.
    embedding_net.apply(weight_initializer)

    # Build density estimator ----------------------------------------------
    # The default density estimator has 3 hidden layers with a number of neurons = hidden_features.
    # The weights are initialized with the default initialization provided by pytorch.
    neural_posterior = utils.posterior_nn(
        model=config["density_estimator"]["type"],
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_components=config["density_estimator"]["args"]["num_components"],
        device=device,
    )

    # Load the training dataset ----------------------------------------------------------
    logger.info("Loading the training dataset...")
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
        (len(dataset), input_shape[0], input_shape[1], input_shape[2])
    )
    for i, (x, theta) in enumerate(dataset):
        # Re-shape the matrix to have the channel number at the beginning.
        x = np.moveaxis(x, -1, 0)
        matrix[i] = x[None, :]
        parameter[i] = theta

    # Transform the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)

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
        prior=prior, density_estimator=neural_posterior, device=f"{device}"
    )

    # Train the network --------------------------------------------------------------------
    logger.info("Train the density estimator...")
    density_estimator = inference.append_simulations(
        parameter.to(device), matrix.to(device), proposal=prior
    ).train(
        learning_rate=config["trainer"]["lr"],
        training_batch_size=config["trainer"]["batch_size"],
        validation_fraction=config["trainer"]["validation_fraction"],
        show_train_summary=True,
    )

    # Save the trained model and statistics -------------------------------------------------
    logger.info("Save the trained model and training statistics...")
    # Extract tensorboard information about the training and validation losses and other training info.
    all_event_data = tbo._get_event_data_from_log_dir(
        inference._summary_writer.log_dir
    )
    scalars = all_event_data["scalars"]

    with open(f"{config.save_dir}/trained_model.pickle", "wb") as output_file:
        pickle.dump(density_estimator.cpu(), output_file)

    training_statistics_path = f"{config.log_dir}/training_statistics.json"
    with open(training_statistics_path, "w") as f:
        json.dump(scalars, f, indent=4, sort_keys=True)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"Accuracy")
    ax.plot(
        scalars["training_log_probs"]["step"],
        scalars["training_log_probs"]["value"],
        linestyle="-",
        linewidth=4,
        color="tab:blue",
        rasterized=True,
        label="training",
    )
    ax.plot(
        scalars["validation_log_probs"]["step"],
        scalars["validation_log_probs"]["value"],
        linestyle="-",
        linewidth=4,
        color="tab:orange",
        rasterized=True,
        label="validation",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), frameon=False, loc=0, fontsize=10)

    f.savefig(f"{config.log_dir}/training_stats.pdf", bbox_inches="tight")


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="Simulation Based Inference Learning"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="examples/learning/config_sbi.json",
        help="Configuration file path.",
    )

    args.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="Path to checkpoint to resume training. This argument is not used at the moment.",
    )

    args.add_argument(
        "--infer",
        nargs="?",
        type=str,
        default=False,
        help="Flag to setup the inference saving path.",
    )

    CustomArgs = collections.namedtuple(
        "CustomArgs", "flags type nargs target"
    )

    options = [
        CustomArgs(
            ["--dataset_training"],
            type=str,
            nargs="?",
            target="training_data_loader;dataset_path",
        ),
        CustomArgs(
            ["--dataset_statistics"],
            type=str,
            nargs="?",
            target="training_data_loader;statistic_path",
        ),
        CustomArgs(
            ["--filter_inputs"],
            type=int,
            nargs="*",
            target="training_data_loader;filter_inputs",
        ),
        CustomArgs(
            ["--filter_labels"],
            type=int,
            nargs="*",
            target="training_data_loader;filter_labels",
        ),
        CustomArgs(
            ["--batch_size"],
            type=int,
            nargs="?",
            target="training_data_loader;batch_size",
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
            ["--save_dir"],
            type=str,
            nargs="?",
            target="trainer;save_dir",
        ),
        CustomArgs(
            ["--normalize"],
            type=bool,
            nargs="?",
            target="training_data_loader;normalize",
        ),
        CustomArgs(
            ["--standardize"],
            type=bool,
            nargs="?",
            target="training_data_loader;standardize",
        ),
        CustomArgs(["--lr"], type=float, nargs="?", target="trainer;lr"),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        args, options
    )

    train(configuration)
