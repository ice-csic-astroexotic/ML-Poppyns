"""
    Inference ensemble script.

    This script performs inference on a test dataset using an ensemble of trained methods specified in the
    args.trained_model txt file. We create our own ensemble approach to ensure it is sufficiently flexible to
    deal with different training experiments. Note that the ensemble method in the sbi package has limitations
    on the types of experiments it can support (e.g., different input shapes).
    Simulation-based Calibration is also performed to check if the ensemble posterior is well behaved.
    See [https://www.mackelab.org/sbi/](https://www.mackelab.org/sbi/) for more details.

    Display help message to run the code:

    python infer_sbi_ensemble.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import collections
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from sbi import utils
from sbi.inference import SNPE

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import utilities.benchmark.timewith as timewith
from pypopsyn.learning.utils.request_device import request_device
from utilities.coverage_probability import coverage_prob


def calculate_smallest_hdr_ensemble(
    experiments: dict,
    posterior_ensemble: callable,
    true_value: torch.tensor,
    posterior_samples_std: torch.tensor,
    posterior_samples_norm: torch.tensor,
    simulation_output: torch.tensor,
    device: str,
) -> float:

    """
    Calculating the smallest highest density region of the posterior ensemble, that contains the true value.

    Args:
        experiments (Dictionary): Dictionary containing the parameters, matrix, posterior and type of scaling
            (std or norm) for each experiment.
        posterior_ensemble (Callable): Ensemble posterior distribution function.
        true_value (torch.tensor): Tensor containing the values of the parameters used to generate the simulated
            population in simulation_output.
        posterior_samples_std (torch.tensor): Tensor containing the samples standardized from the inferred ensemble
            posterior distribution for simulation_output.
        posterior_samples_norm (torch.tensor): Tensor containing the samples normalized from the inferred ensemble
            posterior distribution for simulation_output.
        simulation_output (torch.tensor): Tensor containing the maps of the simulated population.
        device (str): String specifying the type of the device used to run the script.

    Returns:
        hdr (float): Smallest highest density region of the posterior ensemble that contains the true value.
    """

    log_prob_true_value = []
    log_prob_samples = []
    num_components = len(posterior_ensemble)

    # Setting the weights of each component to 1/N, with N denoting the number of ensemble components,
    # because each of the components has the same importance.
    weights = torch.tensor(
        [1.0 / num_components for _ in range(num_components)]
    ).to(device)

    for index, posterior in enumerate(posterior_ensemble):

        # Evaluating the PDF value of the ground truth.
        log_prob_true_value.append(
            posterior.log_prob(
                true_value[index].to(device),
                simulation_output[index].to(device),
            )
        )

        # Evaluating the average log-probability of the posterior ensemble for each of the posterior samples
        # and for the true value.

        # Use the standardized or normalized samples depending on which experiment we are using.
        if experiments[index]["normalize"]:
            posterior_samples = posterior_samples_norm
        else:
            posterior_samples = posterior_samples_std

        log_prob_samples.append(
            posterior.log_prob(
                posterior_samples.to(device),
                simulation_output[index].to(device),
            )
        )

    # Computing the log probability for the ensemble, i.e., the average log probabilities for all samples from all
    # experiments.

    log_prob_true_value = torch.stack(log_prob_true_value)
    log_prob_samples = torch.stack(log_prob_samples)

    log_weights = torch.log(weights).reshape(-1, 1)

    log_prob_true_value = torch.logsumexp(
        log_weights.expand_as(log_prob_true_value) + log_prob_true_value, dim=0
    )
    log_prob_samples = torch.logsumexp(
        log_weights.expand_as(log_prob_samples) + log_prob_samples, dim=0
    )

    # Determining the fraction of average log-probabilities values that are larger than that of the ground truth.
    hdr = (log_prob_samples > log_prob_true_value).sum()
    hdr = hdr / len(log_prob_samples)

    return hdr


def infer(
    args: argparse.Namespace, config: configuration_parser.ConfigurationParser
) -> None:
    """
    Perform simulation-based inference using an ensemble method on different trained models on the provided dataset.

    Args:
        args (argparse.Namespace): Command line arguments containing configuration options:

            - configuration (str): Path to the configuration file.
            - corner_plot (bool): If set to True, generates posterior corner plots for each test sample.
            - trained_model (str): Path to a `.txt` file containing the paths to the pretrained models.
            - infer (str): Flag to set up the inference saving path (default is True).

        config (configuration_parser.ConfigurationParser): Configuration object with settings
            for data loading, model architecture, profiling options, and other parameters.
    """

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

    # Remove the profile.json and profile.log files to prevent interrupted server connection issues.
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

            # Load the trained models from the txt file.
            logger.info("Loading the trained models...")
            logger.info(
                "Inference is performed with the trained models in: {}".format(
                    args.trained_model
                )
            )
            # Opening the txt file with all the path to the trained models.
            with open(args.trained_model, "rb") as f:
                trained_models_path = f.readlines()

            # Initialize the torch seed.
            if config["set_manual_seed"] is True:
                torch.manual_seed(config["manual_seed"])
                logger.info("Seed: {}".format(config["manual_seed"]))
            else:
                torch.manual_seed(int(time.time()))
                logger.info("Seed: {}".format(int(time.time())))

            # Initializing a dictionary and list to store information about each experiment's configuration and its
            # corresponding inference model.

            experiments = {}
            posterior_ensemble = []

            with timewith.TimeWith(
                "[LoadingExperiments]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):
                # Loop through all the different experiments.
                for index, exp in enumerate(trained_models_path):

                    # Load the test dataset ----------------------------------------------------------
                    logger.info("Experiment {}".format(index))

                    # Extracting the path of the trained model and the config associated to each experiment.
                    model_path = exp.strip().split()[0]
                    config_path = exp.strip().split()[1]
                    with open(model_path, "rb") as f:
                        trained_model = pickle.load(f)
                    with open(config_path, "rb") as f_config:
                        config_json = json.load(f_config)

                    dataset_path = config_json["test_data_loader"][
                        "dataset_path"
                    ]
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
                    standardize = config_json["test_data_loader"][
                        "standardize"
                    ]
                    input_shape = config_json["arch"]["args"]["input_shape"]

                    n_parameters = len(filter_labels)

                    logger.info(
                        "Loading the test dataset, applying either norm or std based on the experiment, and saving the rescaled dataset for each experiment..."
                    )
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
                        # Reshape the matrix to have the channel number at the beginning.
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
                    with timewith.TimeWith(
                        "[InferenceSetup]",
                        prof_log_path,
                        prof_json_path,
                        config["show_profiling"],
                    ):
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
                        logger.info(
                            "Loading the amortized trained posterior..."
                        )

                        # By default the procedure uses SNPE-C
                        # (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
                        inference = SNPE()

                        # Building the inferred posterior distribution for each experiment.
                        inference_model = inference.build_posterior(
                            trained_model.to(device), prior=prior
                        )

                        # Store experiment information in the dictionary.
                        experiments[index] = {
                            "normalize": normalize,
                            "posterior": inference_model,
                            "parameter": parameter,
                            "matrix": matrix,
                        }

                        posterior_ensemble.append(inference_model)

        with timewith.TimeWith(
            "[Inference]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):
            logger.info("Computing the coverage probability...")

            hdr_testset = np.zeros(len(dataset))

            for i in range(len(dataset)):

                # The number of posterior samples for each experiment should be small, as we need to calculate the log-
                # probability for each sample in the ensemble, which is a concatenation of samples from all experiments.
                n_samples_coverage_exp = 100

                # To calculate the coverage probability, we normalize and standardize the ensemble samples.
                posterior_samples_coverage_norm = []
                posterior_samples_coverage_std = []
                parameter_sample = []
                matrix_sample = []

                for experiment_name, experiment in experiments.items():

                    parameter = experiment["parameter"]
                    matrix = experiment["matrix"]
                    posterior = experiment["posterior"]

                    posterior_sample_exp = (
                        posterior.set_default_x(matrix[i])
                        .sample(
                            (n_samples_coverage_exp,), show_progress_bars=False
                        )
                        .to(device)
                    )

                    # Save the statistics for the filtered labels. Here, we assume that the test dataset
                    # is the same for all the experiments.
                    par_max = torch.tensor(dataset.target_max).to(device)
                    par_min = torch.tensor(dataset.target_min).to(device)
                    par_std = torch.tensor(dataset.target_std).to(device)
                    par_mean = torch.tensor(dataset.target_mean).to(device)

                    # To standardize and normalize all the ensemble samples, we first convert
                    # them to their original physical ranges.
                    if experiment["normalize"]:
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
                ).to(device)
                posterior_samples_coverage_norm = torch.cat(
                    posterior_samples_coverage_norm
                ).to(device)

                smallest_hdr = calculate_smallest_hdr_ensemble(
                    experiments,
                    posterior_ensemble,
                    parameter_sample,
                    posterior_samples_coverage_std,
                    posterior_samples_coverage_norm,
                    matrix_sample,
                    device,
                )

                hdr_testset[i] = smallest_hdr

            coverage_prob(hdr_testset, 12, config.log_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="PyPopSyn Simulation-Based Inference"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="pypopsyn/learning/config_sbi.json",
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
