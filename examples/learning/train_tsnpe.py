"""
    Training script for truncated sequential neural posterior estimation following Deistler et al. (2022).
    (https://arxiv.org/abs/2210.04815).

    This script implements the truncated sequential neural posterior estimator using the sbi package. It iteratively
    trains a density estimator for `num_rounds`, where each iteration involves generating training and testing datasets
    based on the previously approximated posterior distribution at the observed sample. This approach focuses on
    the region of the parameter space that matches the observed population to save computational resources.

    To create the training and test datasets, we use either the `multiprocessing` or `Dask` (https://www.dask.org/) package
    to run the simulations simultaneously in a multithreaded manner. To use Dask change the variable `enable_dask` in
    the configuration file to True. Otherwise, change it to False to use multiprocessing.

    For further details, visit https://www.mackelab.org/sbi/.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import collections
import pathlib
import pickle
import subprocess
import sys
import time
from logging import Logger
from typing import Optional, Tuple

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import utils
from sbi.inference import SNPE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from tqdm import tqdm

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device
from pypopsyn.simulator.configuration import cfg
from scripts.coverage_probability import coverage_prob
from scripts.simulation_helper_sbi import (
    initialize_dask_cluster,
    simulator_dask,
    simulator_multiprocess,
)


def calculate_smallest_hdr(
    posterior: DirectPosterior,
    theta: torch.tensor,
    matrix: torch.tensor,
    n_samples_coverage: int,
    device: torch.device,
) -> np.ndarray:

    """
    Calculating the smallest highest density region of the posterior distribution, that contains the true value for the
    test dataset produced with the values theta and simulation output in matrix.

    Args:
        posterior (DirectPosterior): Posterior distribution.
        theta (torch.tensor): Tensor containing the values of the parameters used to generate the simulated
                              population in matrix.
        matrix (torch.tensor): Tensor containing the maps of the simulated population.
        n_samples_coverage (float): Number of approximate posterior samples used for computing the coverage.
        device (torch.device): Device used to run the script.

    Returns:
        np.ndarray: Smallest highest density region of the posterior that contains the true value.
    """
    hdr = np.zeros(len(theta))

    for index in tqdm(
        range(theta.size(0)), desc="Computing Coverage Probability"
    ):
        simulation_output = matrix[index]
        true_value = theta[index]
        posterior_samples = posterior.set_default_x(simulation_output).sample(
            (n_samples_coverage,), show_progress_bars=False
        )
        # Evaluating the PDF value of the ground truth.
        log_p_true = posterior.log_prob(
            true_value.to(device), simulation_output.to(device)
        )
        # Evaluating the PDF values of the posterior samples.
        log_p_samples = posterior.log_prob(
            posterior_samples.to(device), simulation_output.to(device)
        )

        # Determining the fraction of PDF values that are larger than that of the ground truth.
        hdr[index] = (log_p_samples > log_p_true).float().mean()

    return hdr


def build_network(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
) -> SNPE_C:
    """
    Building the neural network using the configuration file specified in the arguments.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the neural network
                                                           architecture and other settings.
        device (torch.device): Device used to run the script.

    Returns:
        inference (sbi.inference.snpe.snpe_c.SNPE_C): An instance of sbi's SNPE inference objects.
    """

    embedding_net = config.init_object("arch", learning_models)

    # Initialize weights.
    weight_initializer = config.init_object(
        "weights_initializer", learning_initializers
    )
    # Apply the weight initialization scheme to every layer in the model.
    embedding_net.apply(weight_initializer)

    # Build density estimator.
    # The default density estimator has 3 hidden layers with a number of neurons = hidden_features.
    # The weights are initialized with the default initialization provided by pytorch.
    hidden_features = config["arch"]["args"]["len_output_layer"]

    neural_posterior = utils.posterior_nn(
        model=config["density_estimator"]["type"],
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_components=config["density_estimator"]["args"]["num_components"],
        device=device,
    )

    # Setting up the inference procedure.
    # We use the default option SNPE-C (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
    inference = SNPE(
        density_estimator=neural_posterior,
        device=f"{device}",
    )

    return inference


def wrapper_pypopsyn(
    proposal: DirectPosterior,
    num_sim: int,
    config: configuration_parser.ConfigurationParser,
    round_current: int,
    test: bool,
    dataset: dl.DatasetMultichannelArray,
    device: torch.device,
) -> str:
    """
    Simulating `num_sim` of mock neutron star population given the `proposal` distribution. After simulating the
    populations, we generate the compressed representations for the output.

    Args:
        proposal (DirectPosterior): Proposal distribution used for sampling the parameters.
        num_sim (int): Number of simulations to perform.
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        round_current (int): Number of current round during the sequential inference approach.
        test (bool): Flag indicating whether the simulations are for testing or training. If set to True, the
                     simulations are for testing purposes.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.
        device (torch.device): Device used to run the script.

    Returns:
        str: Path to the generated dataset.
    """

    # Setting paths.
    # If 'test' is True, simulations are saved in the folder specified for the testing dataset in the config file.
    # Otherwise, simulations are saved in the folder specified for the training dataset in the config file.
    if test:
        sim_dir_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/simulations/round_{round_current}"
        )
        dataset_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{round_current}"
        )
    else:
        sim_dir_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/simulations/round_{round_current}"
        )
        dataset_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{round_current}"
        )

    # Extracting simulation parameters from configuration file.
    dyn_data_path = config["dyn_data_loader"]["dataset_path"]
    args_dict = {
        "dyn_data": dyn_data_path,
        "output_dir": sim_dir_path,
        "simulator_type": "simulate_population_magrot_det",
        "sampling_size": num_sim,
        "P_initial_log10_mean": [
            config["prior_ranges"]["low"][2],
            config["prior_ranges"]["high"][2],
        ],
        "P_initial_log10_sigma": [
            config["prior_ranges"]["low"][3],
            config["prior_ranges"]["high"][3],
        ],
        "B_initial_log10_mean": [
            config["prior_ranges"]["low"][0],
            config["prior_ranges"]["high"][0],
        ],
        "B_initial_log10_sigma": [
            config["prior_ranges"]["low"][1],
            config["prior_ranges"]["high"][1],
        ],
        "a_late": [
            config["prior_ranges"]["low"][4],
            config["prior_ranges"]["high"][4],
        ],
        "processes": config["n_processes"],
    }

    # Constructing the command to generate the density map of the simulations.
    software_path = cfg["path_to_software"]
    command = [
        "python",
        f"{software_path}examples/generator/generate_dataset_surveys.py",
        "--data",
        str(sim_dir_path),
        "--save_dir",
        str(dataset_path),
        "--resolution_ppdot",
        str(config["arch"]["args"]["input_shape"][1]),
        "--data_type",
        "array",
    ]

    # Running the simulations and generating the corresponding density maps for each simulation. The simulations are run
    # in a multithreaded manner. If config["enable_dask"] is equal to True, then multithreading will be performed with
    # the Dask package. Otherwise, it will be performed with the multiprocessing package.

    if config["enable_dask"]:
        simulator_dask(args_dict, proposal, dataset, device)
    else:
        simulator_multiprocess(args_dict, proposal, dataset)

    subprocess.run(command)

    return dataset_path


def corner_plot(
    observed_samples: torch.tensor,
    dataset: dl.DatasetMultichannelArray,
    save_dir: str,
) -> None:
    """
    Plotting the corner plot for the posterior distribution.

    Args:
        observed_samples (torch.tensor): Samples of the distribution to plot.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.
        save_dir (str): Directory to save the corner plot.

    Returns:
        None
    """

    # Save the statistics for the filtered labels.
    par_max = torch.tensor(dataset.target_max)
    par_min = torch.tensor(dataset.target_min)
    par_std = torch.tensor(dataset.target_std)
    par_mean = torch.tensor(dataset.target_mean)

    # If the parameters were normalized or standardized rescale quantities to their physical ranges.
    if dataset.normalize:
        observed_samples = observed_samples * (par_max - par_min) + par_min

    elif dataset.standardize:
        observed_samples = observed_samples * par_std + par_mean

    # Saving the best estimated parameters and the 95% CI into the log.txt file.
    quantile = np.quantile(observed_samples, [0.025, 0.5, 0.975], axis=0)

    range_param = [[par_min[v], par_max[v]] for v in range(len(par_max))]

    param_median = quantile[1, :]

    # Corner plot of the inferred posterior distributions for each parameter.
    figure = corner.corner(
        observed_samples.detach().cpu().numpy(),
        bins=32,
        labels=dataset.target_names,
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
    corner.overplot_lines(figure, param_median, color="tab:red")

    corner.overplot_points(
        figure,
        param_median[None],
        marker="s",
        color="tab:red",
    )
    plt.savefig(save_dir)
    plt.close()


def prepare_dataset_sbi(
    train_data_set: str,
    config: configuration_parser.ConfigurationParser,
    logger: Logger,
    atnf: Optional[bool] = False,
) -> Tuple[dl.DatasetMultichannelArray, torch.tensor, torch.tensor]:
    """
    Prepare dataset for use in sbi training.

    Args:
        train_data_set (str): Path to the training dataset.
        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.
        atnf (bool, optional): Indicates whether the PPdot density maps in the 'train_data_set' folder correspond to
                               the observed population or to a simulated population. If set to True, the simulations
                               correspond to the observed ATNF population. The default is False.
        logger (Logger): Logger object.

    Returns:
        tuple: A tuple containing the dataset, parameter tensor and input matrix tensor.
    """

    # Adjusting the dataset_path based on whether the dataset is the observed or a simulated population.
    dataset_path = (
        train_data_set + "/dataset_atnf.csv"
        if atnf
        else train_data_set + "/dataset_full.csv"
    )
    dataset_stat_path = config["training_data_loader"]["statistic_path"]
    filter_inputs = config["training_data_loader"]["filter_inputs"]
    filter_labels = config["training_data_loader"]["filter_labels"]
    normalize = config["training_data_loader"]["normalize"]
    standardize = config["training_data_loader"]["standardize"]
    input_shape = config["arch"]["args"]["input_shape"]
    n_parameters = len(filter_labels)

    # Loading the training dataset.
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
        logger.exception("Error: an error occurred when loading the dataset.")
        sys.exit(1)

    parameter = np.zeros((len(dataset), n_parameters))
    matrix = np.zeros(
        (len(dataset), input_shape[0], input_shape[1], input_shape[2])
    )
    for i, (x, theta) in enumerate(dataset):
        # Reshaping the matrix to have the channel number at the beginning.
        x = np.moveaxis(x, -1, 0)

        if list(x.shape) != input_shape:
            logger.error(
                "Mismatch between the shape of the input data x {} and the input shape specified "
                "in the configuration file {}".format(x.shape, input_shape)
            )
            sys.exit()

        matrix[i] = x
        parameter[i] = theta

    # Transforming the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)
    return dataset, parameter, matrix


def train(args, config):
    """
    Training a density estimator to infer the posterior distribution at the observed population with the truncated
    sequential neural posterior estimator approach in Deistler et al. (2022) using the sbi package.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.

    Returns:
        None
    """
    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("train")
    logger.info("Logger initialized...")

    # Initialize the path where the time profiling will be saved.
    prof_log_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_log"])
    )
    prof_json_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_json"])
    )

    # Set up GPU device if available.
    logger.info("Requesting {} GPUs...".format(config["n_gpu"]))
    device, device_ids = request_device(logger, config["n_gpu"])
    logger.info("Devices obtained: {}".format(device_ids))

    if config["enable_dask"]:
        with timewith.TimeWith(
            "[InitializingDask]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):
            logger.info("Initializing dask cluster...")
            cluster = initialize_dask_cluster(logger, config)

    # Show experiment information ------------------------------------------
    logger.info("=========================================================")
    with timewith.TimeWith(
        "[TotalTraining]",
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
            logger.info("Loading the training dataset for the first round...")
            train_dataset_path = config["training_data_loader"][
                "dataset_path_first_round"
            ]

            logger.info(
                "Preparing the training data set for sbi for the first round..."
            )
            dataset, parameter, matrix = prepare_dataset_sbi(
                train_dataset_path, config, logger
            )

            n_parameters = len(torch.tensor(config["prior_ranges"]["low"]))
            num_rounds = config["trainer"]["num_rounds"]

            if config["set_manual_seed"] is True:
                torch.manual_seed(config["manual_seed"])
                logger.info("Seed: {}".format(config["manual_seed"]))
            else:
                torch.manual_seed(int(time.time()))
                logger.info("Seed: {}".format(int(time.time())))

            logger.info("Defining the prior distribution...")

            # Setting the prior distribution for the parameters.
            # Note that we need to rescale the prior distribution to ensure that it has the correct limits
            # when restricted.
            if config["training_data_loader"]["normalize"]:
                # All the parameters are rescaled in the range [0, 1].
                prior = utils.BoxUniform(
                    low=torch.tensor(np.zeros(n_parameters)),
                    high=torch.tensor(np.ones(n_parameters)),
                    device=f"{device}",
                )
            elif config["training_data_loader"]["standardize"]:
                low = (
                    torch.tensor(config["prior_ranges"]["low"])
                    - dataset.target_mean
                ) / dataset.target_std
                high = (
                    torch.tensor(config["prior_ranges"]["high"])
                    - dataset.target_mean
                ) / dataset.target_std
                prior = utils.BoxUniform(
                    low=low,
                    high=high,
                    device=f"{device}",
                )
            else:
                # Set the prior range to the range of the parameters.
                prior = utils.BoxUniform(
                    low=torch.tensor(config["prior_ranges"]["low"]),
                    high=torch.tensor(config["prior_ranges"]["high"]),
                    device=f"{device}",
                )

            # At the first round, we set the proposal equal to the prior.
            proposal = prior

            # Lists to store parameters and matrices from each round.
            parameter_list = []
            matrix_list = []

            logger.info("Building the neural network...")
            inference = build_network(config, device)

        for i in range(num_rounds):
            with timewith.TimeWith(
                f"[TotalRound{i}]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):
                with timewith.TimeWith(
                    f"[TrainingRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):

                    num_sim_train = config["training_data_loader"]["num_sim"]

                    # Creating a folder to save the model, coverage and posterior distribution for each round.
                    save_dir_round = config.save_dir / f"round_{i}"
                    save_dir_round.mkdir(parents=True, exist_ok=True)

                    # In the first round, instead of simulating the training dataset, we use the simulations
                    # previously run.
                    if i > 0:
                        logger.info(
                            f"Training, ------------------------------- round {i} -------------------------------------"
                        )

                        logger.info(
                            "Simulating the training dataset for {} simulations...".format(
                                num_sim_train
                            )
                        )
                        train_dataset_path = wrapper_pypopsyn(
                            proposal,
                            config=config,
                            num_sim=num_sim_train,
                            round_current=i,
                            test=False,
                            dataset=dataset,
                            device=device,
                        )

                        # Building the training dataset for sbi.
                        logger.info(
                            "Preparing the training data set for sbi..."
                        )
                        dataset, parameter, matrix = prepare_dataset_sbi(
                            train_dataset_path, config, logger
                        )

                    # Saving the training data to reuse it in the next rounds.
                    parameter_list.append(parameter)
                    matrix_list.append(matrix)
                    parameter_round = torch.cat(parameter_list, dim=0)
                    matrix_round = torch.cat(matrix_list, dim=0)

                    logger.info(
                        f"Training the density estimator with {parameter_round.shape[0]} samples in round {i} ..."
                    )

                    # If config["ensemble"] is set to 'True', instead of training a single neural network,
                    # 'config["size_ensemble"]' neural networks are trained to construct an ensemble of posteriors.
                    # This will prevent narrow posteriors.
                    if config["trainer"]["ensemble"]:
                        size_ensemble = config["trainer"]["size_ensemble"]

                        logger.info(
                            f"Training an ensemble of {size_ensemble} mixture density networks..."
                        )
                        posteriors_list = []

                        for index in range(size_ensemble):
                            # Training each of the networks that will create the ensemble.
                            density_estimator = inference.append_simulations(
                                parameter_round.to(device),
                                matrix_round.to(device),
                            ).train(
                                learning_rate=config["trainer"]["lr"],
                                training_batch_size=config["trainer"][
                                    "batch_size"
                                ],
                                validation_fraction=config["trainer"][
                                    "validation_fraction"
                                ],
                                show_train_summary=True,
                                force_first_round_loss=True,
                            )
                            # Build the posterior object for each trained network.
                            posterior_ensemble = inference.build_posterior(
                                density_estimator.to(device), prior=prior
                            )
                            logger.info(
                                f"Saving the trained model for round {i}..."
                            )

                            with open(
                                f"{save_dir_round}/trained_model_ensemble_{index}.pickle",
                                "wb",
                            ) as output_file:
                                pickle.dump(
                                    density_estimator.cpu(), output_file
                                )

                            posteriors_list.append(posterior_ensemble)
                        # Setting the weights of each ensemble posterior directly to enable the use of a GPU.
                        weights_ensemble = (
                            torch.ones(size_ensemble) / size_ensemble
                        )
                        # Build the ensemble using the trained neural networks.
                        posterior = NeuralPosteriorEnsemble(
                            posteriors_list,
                            weights=weights_ensemble.to(device),
                        )

                    else:
                        # If config["ensemble"] is set to False, only a single neural network will be trained to
                        # approximate the posterior distribution.
                        density_estimator = inference.append_simulations(
                            parameter_round.to(device), matrix_round.to(device)
                        ).train(
                            learning_rate=config["trainer"]["lr"],
                            training_batch_size=config["trainer"][
                                "batch_size"
                            ],
                            validation_fraction=config["trainer"][
                                "validation_fraction"
                            ],
                            show_train_summary=True,
                            force_first_round_loss=True,
                        )

                        with open(
                            f"{save_dir_round}/trained_model.pickle",
                            "wb",
                        ) as output_file:
                            pickle.dump(density_estimator.cpu(), output_file)

                        posterior = inference.build_posterior(
                            density_estimator.to(device), prior=prior
                        )

                with timewith.TimeWith(
                    f"[TestingRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):

                    num_sim_test = config["test_data_loader"]["num_sim"]

                    if i == 0:
                        logger.info(
                            "Loading the test dataset for the first round..."
                        )
                        test_dataset_path = config["test_data_loader"][
                            "dataset_path_first_round"
                        ]

                    else:
                        logger.info(
                            "Simulating the test dataset for {} simulations...".format(
                                num_sim_test
                            )
                        )
                        test_dataset_path = wrapper_pypopsyn(
                            proposal,
                            config=config,
                            num_sim=num_sim_test,
                            round_current=i,
                            test=True,
                            dataset=dataset,
                            device=device,
                        )

                    _, parameter_test, matrix_test = prepare_dataset_sbi(
                        test_dataset_path, config, logger
                    )

                    logger.info(
                        f"Computing coverage probability for the test dataset for round {i}..."
                    )
                    num_posterior_samples = 1000
                    hdr = calculate_smallest_hdr(
                        posterior,
                        parameter_test,
                        matrix_test,
                        num_posterior_samples,
                        device=device,
                    )

                    coverage_prob(hdr, n_betas=12, save_dir=save_dir_round)

                with timewith.TimeWith(
                    f"[ComputeRestrictedPriorRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    logger.info(
                        f"Computing the proposal prior for round {i + 1}..."
                    )
                    # Create the matrix for the observed sample of neutron stars.
                    _, _, x_o = prepare_dataset_sbi(
                        config["observed_sample"]["dataset_path"],
                        config,
                        logger,
                        atnf=True,
                    )
                    # Computing the region of the posterior distribution used to constrain the prior.
                    posterior_obs = posterior.set_default_x(x_o)
                    accept_reject_fn = utils.get_density_thresholder(
                        posterior_obs,
                        quantile=1e-4,
                        num_samples_to_estimate_support=10000,
                    )
                    # Computing the new proposal by restricting the prior to the posterior of the observation.
                    # If config["sir"] is set to true, the restricted prior sampling uses sampling importance
                    # resampling (Rubin et al., 1988); otherwise, it employs rejection sampling. Note that the latter
                    # method may take longer for a narrowed posterior distribution where the rejection rate is high.
                    if config["trainer"]["sir"]:
                        proposal = utils.RestrictedPrior(
                            prior,
                            accept_reject_fn,
                            posterior=posterior_obs,
                            sample_with="sir",
                            device=f"{device}",
                        )
                    else:
                        proposal = utils.RestrictedPrior(
                            prior,
                            accept_reject_fn,
                            sample_with="rejection",
                            device=f"{device}",
                        )

                    if args.plot_proposal:
                        # If `args.plot_proposal` is set to True, a corner plot of the proposal distribution will be
                        # produced. Note that this might take a while since we are using SIR or rejection methods to
                        # sample from the proposal distribution.
                        observed_samples_proposal = proposal.sample(
                            (50000,), show_progress_bars=False
                        ).cpu()
                        corner_plot(
                            observed_samples_proposal,
                            dataset,
                            f"{save_dir_round}/corner_plot_prior_round_{i+1}.pdf",
                        )
                        # Save the samples from the inferred posterior distribution.
                        torch.save(
                            observed_samples_proposal,
                            f"{save_dir_round}/samples_prior_{i+1}.pt",
                        )

                logger.info(f"Saving the trained model for round {i}...")

                with open(
                    f"{save_dir_round}/trained_model_{i}.pickle", "wb"
                ) as output_file:
                    pickle.dump(density_estimator.cpu(), output_file)

                logger.info(
                    f"Inferring the parameters for the observed sample for round {i}..."
                )

                observed_samples_posterior = posterior_obs.sample(
                    (50000,), show_progress_bars=False
                ).cpu()
                corner_plot(
                    observed_samples_posterior,
                    dataset,
                    f"{save_dir_round}/corner_plot_observed_sample_{i}.pdf",
                )
                torch.save(
                    observed_samples_posterior,
                    f"{save_dir_round}/samples_posterior_{i}.pt",
                )

        if config["enable_dask"]:
            # Closing the cluster once the training has finished.
            cluster.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Truncated SNPE trainer")

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="examples/learning/config_tsnpe.json",
        help="Machine learning configuration file path.",
    )

    args.add_argument(
        "--plot_proposal",
        type=bool,
        default=False,
        help="If the proposal corner plot for each round is required, this argument should be set to True.",
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
        help="Flag to setup the inference saving path. This argument is not used at the moment.",
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

    configuration = configuration_parser.ConfigurationParser.from_args(args)

    train(args.parse_args(), configuration)
