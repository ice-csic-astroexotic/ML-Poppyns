"""
    Sbi utilities.

    Display help message to run the code:

    python sbi_utils.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import json
import os
import pathlib
import pickle
import signal
import sys
from logging import Logger
from types import FrameType
from typing import List, Optional, Tuple, Union

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.analysis import check_sbc, run_sbc, sbc_rank_plot
from sbi.analysis import tensorboard_output as tbo
from sbi.inference import SNLE, SNPE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snle.snle_a import SNLE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils import BoxUniform
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from tqdm import tqdm

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loader_multichannel_array as dl
import pypopsyn.learning.models.models as learning_models
import utilities.benchmark.timewith as timewith
from pypopsyn.generator import generate_dataset_surveys
from utilities.coverage_probability import coverage_prob
from utilities.experiment_helpers.run_simulation_set_sbi import (
    simulator_dask,
    simulator_multiprocess,
)


def handler(signum: int, frame: FrameType) -> None:
    """
    Signal handler that raises a TimeoutError when a SIGALRM signal is received.  This function is needed in the
    sample_with_timeout function to raise an error if the sampling exceeds the maximum time.

    Args:
        signum (int): The signal number.
        frame (Any): The current stack frame.

    Raises:
        (TimeoutError): Indicates that the operation timed out.
    """
    raise TimeoutError("The operation timed out")


def sample(
    posterior: DirectPosterior,
    simulation_output: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
     Perform sampling from the posterior distribution. This function is designed to be used with a signal handler
     to enforce a timeout during sampling, see below.

     Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples (int): The number of samples to draw from the posterior distribution.

    Returns:
        (torch.Tensor): The samples drawn from the posterior distribution.
    """
    return posterior.set_default_x(simulation_output).sample(
        (n_samples,), show_progress_bars=False
    )


def sample_with_timeout(
    posterior: DirectPosterior,
    simulation_output: torch.Tensor,
    n_samples: int,
    timeout: Optional[int] = 180,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Perform sampling from the posterior distribution with a specified timeout.

    Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples (int): The number of samples to draw from the posterior distribution.
        timeout (int, optional): The maximum time in seconds to allow for n_samples to be drawn. Defaults to 180 seconds.

    Returns:
        (Tuple[Optional[torch.Tensor], bool]): A tuple containing the result of the sampling (or None if it times out)
            and a boolean indicating whether the sampling was successful.
    """
    # Set the signal handler and start the alarm to enforce the function timeout.
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        # If the sampling completes before the timeout, return the result and set success to True.
        result = sample(posterior, simulation_output, n_samples)
        success = True
    except TimeoutError:
        # If the timeout is reached, return None and set success to False.
        result = None
        success = False
    finally:
        # Cancel the timer whether an exception was raised or not to prevent the signal from being sent after
        # completion.
        signal.alarm(0)

    return result, success


def calculate_smallest_hdr(
    posterior: DirectPosterior,
    theta: torch.tensor,
    matrix: torch.tensor,
    n_samples: int,
    logger: Logger,
    device: torch.device,
) -> np.ndarray:
    """
    Calculating the smallest highest density region of the posterior distribution, that contains the true value for the
    test dataset produced with the ground truths and simulation output stored in the arguments theta and matrix,
    respectively.

    Args:
        posterior (DirectPosterior): Posterior distribution.
        theta (torch.tensor): Tensor containing the values of the parameters used to generate the simulated
            population in matrix.
        matrix (torch.tensor): Tensor containing the maps of the simulated population.
        n_samples (float): Number of approximate posterior samples.
        logger (Logger): Logger object.
        device (torch.device): Device used to run the script.

    Returns:
        (np.ndarray): Smallest highest density region of the posterior that contains the true value.
    """
    hdr = []
    # Counter for successful samples.
    successful_samples = 0

    for index in tqdm(
        range(theta.size(0)), desc="Computing Coverage Probability"
    ):
        simulation_output = matrix[index]

        # Adding batch dimension (e.g., converting the shape from [3,32,32] to [1,3,32,32]).
        # This is needed for sbi version 0.22.0.
        simulation_output = simulation_output.unsqueeze(0)
        true_value = theta[index]

        # Perform sampling with a timeout of 180 seconds.
        posterior_samples, success = sample_with_timeout(
            posterior, simulation_output, n_samples, timeout=180
        )

        if not success:
            # Skip this test sample if the sampling times out.
            logger.info(
                f"Skipping test sample with index {index} due to timeout."
            )
            continue

        # Increment the successful samples counter.
        successful_samples += 1

        # Evaluating the PDF value of the ground truth.
        log_p_true = posterior.log_prob(
            true_value.to(device), simulation_output.to(device)
        )
        # Evaluating the PDF values of the posterior samples.
        log_p_samples = posterior.log_prob(
            posterior_samples.to(device), simulation_output.to(device)
        )
        # Determining the fraction of PDF values that are larger than that of the ground truth.
        hdr_value = (log_p_samples > log_p_true).float().mean()

        # Handle the device to ensure coverage works on both CPU and GPU.
        if device.type == "cuda":
            hdr_value = hdr_value.cpu().item()
        else:
            hdr_value = hdr_value.item()

        hdr.append(hdr_value)

    percentage_successful = (successful_samples / theta.size(0)) * 100

    # Log the number of successful test samples used to compute the coverage.
    logger.info(
        f"Percentage of successful samples used to compute coverage: {percentage_successful}"
    )

    return np.array(hdr)


def load_inference(
    config: configuration_parser.ConfigurationParser,
    round_number: int,
    save_dir: pathlib.Path,
    ensemble: bool = False,
) -> Union[List[Union[SNPE_C, SNLE_A]], SNPE_C, SNLE_A]:
    """
    Load inference objects from pickle files. Note that this is used when resume mode is enabled or when doing inference
    with snle.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the settings.
        round_number (int): The round number to load the inference from.
        save_dir (pathlib.Path): The directory to load the inference from.
        ensemble (bool): Flag indicating if ensemble mode is enabled. Defaults to False.

    Returns:
        (Union[List[Union[SNPE_C, SNLE_A]], SNPE_C, SNLE_A]): A list of inference objects.
    """
    inference_list = []

    for i in range(config["trainer"]["size_ensemble"] if ensemble else 1):

        inference_path = (
            os.path.join(
                save_dir, f"round_{round_number}/inference_ensemble_{i}.pickle"
            )
            if ensemble
            else os.path.join(
                save_dir, f"round_{round_number}/inference.pickle"
            )
        )

        if not os.path.exists(inference_path):
            raise FileNotFoundError(
                f"The folder specified at {inference_path} in the config file does not contain a inference.pickle file.\n"
                "To use the resume mode or inference with snle, you need to specify the correct path."
            )

        with open(inference_path, "rb") as inference_file:
            inference = pickle.load(inference_file)

        inference_list.append(inference)

    return inference_list


def wrapper_pypopsyn(
    proposal: Union[DirectPosterior, utils.RestrictedPrior],
    num_sim: int,
    config: configuration_parser.ConfigurationParser,
    effective_round: int,
    test: bool,
    dataset: dl.DatasetMultichannelArray,
    device: torch.device,
) -> str:
    """
    Simulating `num_sim` of mock neutron star populations given the `proposal` distribution. After simulating the
    populations, we generate the compressed representations for the output.

    Args:
        proposal (Union[DirectPosterior,utils.RestrictedPrior]): Proposal distribution used for sampling the parameters.
        num_sim (int): Number of simulations to perform.
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        effective_round (int): Number of the effective round during the sequential inference approach. Note that when
            resume is False, the effective round is the same as the actual round. On the other hand, if resume mode is
            enabled, effective_round = last_completed_round + actual_round.
        test (bool): Flag indicating whether the simulations are for testing or training. If set to True, the
            simulations are for testing purposes.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.
        device (torch.device): Device used to run the script.

    Returns:
        (str): Path to the generated dataset.
    """

    # Setting paths.
    # If 'test' is True, simulations are saved in the folder specified for the testing dataset in the config file.
    # Otherwise, simulations are saved in the folder specified for the training dataset in the config file.
    if test:
        sim_dir_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/simulations/round_{effective_round}"
        )
        dataset_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{effective_round}"
        )
    else:
        sim_dir_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/simulations/round_{effective_round}"
        )
        dataset_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{effective_round}"
        )

    # Extracting simulation parameters from configuration file.
    dyn_data_path = config["dyn_data_loader"]["dataset_path"]
    args_dict = {
        "dyn_data": dyn_data_path,
        "save_dir": sim_dir_path,
        "simulator_type": "simulate_population_magrot_det",
        "sampling_size": num_sim,
        "processes": config["n_processes"],
    }

    args_gen = argparse.Namespace(
        data=str(sim_dir_path),
        save_dir=str(dataset_path),
        resolution_ppdot=config["arch"]["args"]["input_shape"][1],
        resolution_dyn=32,
        data_type="array",
    )

    # Running the simulations and generating the corresponding density maps for each simulation. The simulations are run
    # in a multithreaded manner. If config["enable_dask"] is equal to True, then multithreading will be performed with
    # the Dask package. Otherwise, it will be performed with the multiprocessing package.

    if config["enable_dask"]:
        simulator_dask(args_dict, proposal, dataset, device)
    else:
        simulator_multiprocess(args_dict, proposal, dataset, device)

    generate_dataset_surveys.generate_dataset(args_gen)

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


def merge_all_rounds_dataset(
    base_path: pathlib.Path, last_completed_round: int
) -> pathlib.Path:
    """
    Merge all dataset_full.csv files from each round into a single DataFrame. This is necessary in resume mode because,
    during the first round of resuming the training, we need to load all the previous training datasets from the earlier
    rounds.

    Args:
        base_path (Path): The base path where the generated datasets are stored.
        last_completed_round (int): Last completed round number.

    Returns:
        (pathlib.Path): The path to the merged dataset.
    """
    dataframes = []

    # Iterate through the directories to find all the training datasets for each round.
    for i in range(last_completed_round + 1):
        round_path = os.path.join(base_path, f"round_{i}")
        dataset_path = os.path.join(round_path, "dataset_full.csv")

        df = pd.read_csv(dataset_path)
        dataframes.append(df)

    # Concatenating all the training datasets into one to use in the first round of the resumed inference.
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Define the path for the merged dataset.
    output_path = os.path.join(
        base_path, f"combine_round_{last_completed_round + 1}"
    )
    os.makedirs(output_path, exist_ok=True)
    merged_dataset_path = os.path.join(output_path, "dataset_full.csv")

    # Save the merged dataframe to a CSV file.
    merged_df.to_csv(merged_dataset_path, index=False)

    return output_path


def save_training_statistics(
    config: configuration_parser.ConfigurationParser,
    inference: Union[SNPE_C, SNLE_A],
    index: int,
    effective_round: int,
) -> None:
    """
    Save training statistics including scalars and training/validation loss plots.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        inference (Union[SNPE_C, SNLE_A]): sbi inference object.
        index (int): The ensemble index, if ensemble is set to False index is equal to 0.
        effective_round (int): Number of the effective round during the sequential inference approach. Note that when
            resume is False, the effective round is the same as the actual round. On the other hand, if resume mode is
            enabled, effective_round = last_completed_round + actual_round.
    """
    all_event_data = tbo._get_event_data_from_log_dir(
        inference._summary_writer.log_dir
    )
    training_statistics = all_event_data["scalars"]

    log_dir_round_path = os.path.join(
        config.log_dir, f"round_{effective_round}"
    )
    os.makedirs(log_dir_round_path, exist_ok=True)

    training_statistics_path = (
        f"{log_dir_round_path}/training_statistics_{index}.json"
    )
    with open(training_statistics_path, "w") as f:
        json.dump(training_statistics, f, indent=4, sort_keys=True)

    # Save the plot showing the evolution of the training and validation losses.
    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"Accuracy")
    ax.plot(
        training_statistics["training_log_probs"]["step"],
        training_statistics["training_log_probs"]["value"],
        linestyle="-",
        linewidth=4,
        color="tab:blue",
        rasterized=True,
        label="training",
    )
    ax.plot(
        training_statistics["validation_log_probs"]["step"],
        training_statistics["validation_log_probs"]["value"],
        linestyle="-",
        linewidth=4,
        color="tab:orange",
        rasterized=True,
        label="validation",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), frameon=False, loc=0, fontsize=10)

    f.savefig(
        f"{log_dir_round_path}/training_stats_{index}.pdf", bbox_inches="tight"
    )


def compute_proposal_prior(
    posterior_obs: DirectPosterior,
    config: configuration_parser.ConfigurationParser,
    prior: utils.BoxUniform,
    device: torch.device,
) -> utils.RestrictedPrior:
    """
    Compute the proposal prior by restricting the prior to the regions where the posterior of the observation has
        non-negligible mass.
    Args:
        posterior_obs (DirectPosterior): Posterior distribution at the observation.
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        prior (utils.BoxUniform): Prior distribution.
        device (torch.device): Device used for training.

    Returns:
        (utils.RestrictedPrior): The restricted prior based on the posterior distribution of the observation.
    """
    # Computing the region of the posterior distribution used to constrain the prior.
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
    return proposal


def compute_rank_coverage(
    save_dir: pathlib,
    parameter: torch.tensor,
    matrix: torch.tensor,
    posterior: DirectPosterior,
    device: torch.device,
    parameter_labels: List[str],
    logger: Logger,
    effective_round: int,
):
    """
    Compute and visualize ranks and coverage probability for a test dataset.

    Args:
        save_dir (str): Directory to save the computed results and plots.
        parameter (torch.Tensor): Tensor containing the parameters for the test dataset in the current round.
        matrix (torch.Tensor): Tensor containing the matrices for the test dataset in the current round.
        posterior (DirectPosterior): Approximated posterior distribution.
        device (torch.device): Device used for training.
        parameter_labels (List[str]): Labels for the parameters in the test dataset.
        logger (Logger): Logger object.
        effective_round (int): Number of the effective round during the sequential inference approach. Note that when
            resume is False, the effective round is the same as the actual round. On the other hand, if resume mode is
            enabled, effective_round = last_completed_round + actual_round.
    """
    logger.info(
        f"Computing coverage probability for the test dataset for round {effective_round}..."
    )
    num_posterior_samples = 10000
    hdr = calculate_smallest_hdr(
        posterior,
        parameter.to(device),
        matrix.to(device),
        num_posterior_samples,
        logger=logger,
        device=device,
    )

    coverage_prob(hdr, n_betas=12, save_dir=save_dir)

    ranks, dap_samples = run_sbc(
        parameter.to(device),
        matrix.to(device),
        posterior,
        num_posterior_samples=num_posterior_samples,
    )

    # Saving the ranks and the number of posterior samples to reproduce the plot.
    torch.save(ranks, f"{save_dir}/ranks.pt")
    logger.info(
        f"Number of posterior samples to generate the plot of the ranks is {num_posterior_samples}"
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
        num_c2st_repetitions=5,
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
        num_bins=30,  # When passing None the default is len(dataset_test) / 20.
        parameter_labels=parameter_labels,
    )

    f.savefig(
        f"{save_dir}/ranks_histograms.pdf",
        bbox_inches="tight",
    )
    plt.close()
    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="cdf",
        parameter_labels=parameter_labels,
    )

    f.savefig(
        f"{save_dir}/ranks_cumulative.pdf",
        bbox_inches="tight",
    )
    plt.close()


def build_network_snpe(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    prior: utils.BoxUniform,
) -> SNPE_C:
    """
    Building the neural network for SNPE (composed of the embedding net and the density estimator) using the
    configuration file specified in the arguments, and setting up the inference procedure.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the neural network
            architecture and other settings.
        device (torch.device): Device used to run the script.
        prior (utils.BoxUniform): Prior distribution.

    Returns:
        (SNPE_C): An instance of sbi's SNPE inference objects.
    """

    # Building the embedding network.
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
    inference = SNPE(
        density_estimator=neural_posterior,
        device=f"{device}",
        prior=prior,
    )

    return inference


def build_network_snle(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    prior: utils.BoxUniform,
) -> SNLE_A:
    """
    Building inference procedure for SNLE.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the neural network
            architecture and other settings.
        device (torch.device): Device used to run the script.
        prior (utils.BoxUniform): Prior distribution.
    Returns:
        (SNLE_C): An instance of sbi's SNLE inference objects.

    """

    inference = SNLE(
        density_estimator=config["density_estimator"]["type"],
        device=f"{device}",
        prior=prior,
    )

    return inference


def initialize_inference(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    prior: utils.BoxUniform,
    logger: Logger,
    ensemble: bool = False,
) -> Union[List[SNPE], List[SNLE]]:
    """
    Initialize inference objects using the provided configuration.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the network and other settings.
        device (torch.device): Device used to run the script.
        prior (utils.BoxUniform): Prior distribution used for inference.
        logger (Logger): Logger object.
        ensemble (bool): Flag indicating if ensemble mode is enabled.

    Returns:
        (Union[List[SNPE], List[SNLE]]): List of initialized inference objects.
    """
    inference_list = []
    model_type = config["trainer"]["type"]

    for _ in range(config["trainer"]["size_ensemble"] if ensemble else 1):
        if model_type == "snle":
            inference = build_network_snle(config, device, prior)
        elif model_type == "snpe":
            inference = build_network_snpe(config, device, prior)
        else:
            logger.exception(
                "The model type '{}' is not supported. ".format(model_type)
            )
            sys.exit(1)

        inference_list.append(inference)

    return inference_list


def prepare_dataset_sbi(
    dataset_folder: str,
    config: configuration_parser.ConfigurationParser,
    logger: Logger,
    atnf: Optional[bool] = False,
) -> Tuple[dl.DatasetMultichannelArray, torch.tensor, torch.tensor]:
    """
    Prepare dataset for use in sbi.

    Args:
        dataset_folder (str): Path to the folder where the dataset is saved.
        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.
        atnf (bool, optional): Indicates whether the PPdot density maps in the 'train_data_set' folder correspond to
            the observed population or to a simulated population. If set to True, the simulations correspond to the
            observed ATNF population. The default is False.
        logger (Logger): Logger object.

    Returns:
        (tuple): A tuple containing the dataset, parameter tensor and input matrix tensor.
    """

    # Adjusting the dataset_path based on whether the dataset is the observed or a simulated population.
    dataset_path = (
        dataset_folder + "/dataset_atnf.csv"
        if atnf
        else dataset_folder + "/dataset_full.csv"
    )

    dataset_stat_path = config["training_data_loader"]["statistic_path"]

    if atnf:
        filter_inputs = config["observed_sample"]["filter_inputs"]
        filter_labels = config["observed_sample"]["filter_labels"]
    else:
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

    if config["trainer"]["embedding"]:
        matrix = np.zeros(
            (len(dataset), config["arch"]["args"]["len_output_layer"])
        )
        # Load the pre-trained embedding network to encode each dataset sample into a latent vector.
        embedding_model_path = config["trainer"]["trained_embedding"]
        with open(embedding_model_path, "rb") as f:
            emb_neural_net = pickle.load(f)
    else:
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

        if config["trainer"]["embedding"]:
            x_embedded = (
                emb_neural_net._embedding_net(torch.tensor(x)).detach().numpy()
            )
            if normalize:
                matrix[i] = (x_embedded - x_embedded.min()) / (
                    x_embedded.max() - x_embedded.min()
                )
            else:
                matrix[i] = (x_embedded - x_embedded.mean()) / (
                    x_embedded.std() + 1e-8
                )
        else:
            matrix[i] = x

        parameter[i] = theta

    # Transforming the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)
    return dataset, parameter, matrix


def build_posterior(
    config: configuration_parser.ConfigurationParser,
    save_dir_round: pathlib.Path,
    logger: Logger,
    inference_list: Union[List[Union[SNPE_C, SNLE_A]], SNPE_C, SNLE_A],
    parameter_round: torch.Tensor,
    matrix_round: torch.Tensor,
    device: torch.device,
    round_current: int,
    prof_log_path: str,
    prof_json_path: str,
    retrain_from_scratch: bool = False,
    proposal: DirectPosterior = None,
) -> Union[DirectPosterior, NeuralPosteriorEnsemble]:
    """
    Train the density estimator for a given round.

    If resuming is set to True, this mode allows training to continue from the last completed round if interrupted.
    It uses the previously saved state to resume training without starting over.
    If ensemble training is enabled, multiple models (an ensemble) are trained and their predictions are combined to
    ensure conservative coverages. Each of the neural networks will be trained on the same training dataset.

    Note that the inference object should be different for each component of the ensemble to ensure independent weights
    for each component. Moreover, if config['trainer']['model_type'] == 'snle', then an MCMC sampler is needed to sample
    from the posterior distribution.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        save_dir_round (pathlib.Path): Directory where the trained model will be saved or is saved already.
        logger (Logger): Logger object.
        inference_list (Union[List[Union[SNPE_C, SNLE_A]], SNPE_C, SNLE_A]): sbi inference object or list of inference
        objects for ensemble.
        parameter_round (torch.Tensor): Tensor containing the parameters for the current round.
        matrix_round (torch.Tensor): Tensor containing the matrices for the current round.
        device (torch.device): Device used for training.
        round_current (int): Current round number.
        prof_json_path (str): The profile.json path.
        prof_log_path (str): The profile.log path.
        retrain_from_scratch (bool): Whether to retrain the conditional density estimator for the posterior from
            scratch each round. Default value is False.
        proposal (DirectPosterior): The proposal prior use in that round.

    Returns:
        (Union[DirectPosterior, NeuralPosteriorEnsemble]): Trained density estimator or ensemble of estimators.
    """
    ensemble = config["trainer"]["ensemble"]
    ensemble_size = config["trainer"]["size_ensemble"] if ensemble else 1
    resume = config["resume_training"]["resume"]
    last_round = config["resume_training"]["last_round"]
    model_type = config["trainer"]["type"]

    # Determining the number of the effective inference round of the sequential sbi approach. Note that the
    # round_current number is not the effective round when resume mode is enabled, as we did not start from 0.
    effective_round = (
        round_current + int(last_round) if resume else round_current
    )

    posteriors_list = []

    for index in range(ensemble_size):

        if ensemble:
            trained_model_path = os.path.join(
                save_dir_round, f"trained_model_ensemble_{index}.pickle"
            )
            inference_model_path = os.path.join(
                save_dir_round, f"inference_ensemble_{index}.pickle"
            )
        else:
            trained_model_path = os.path.join(
                save_dir_round, "trained_model.pickle"
            )
            inference_model_path = os.path.join(
                save_dir_round, "inference.pickle"
            )

        inference = inference_list[index]

        # If we are in the first round of training and resume mode is enabled, the model is loaded from the last
        # completed round instead of being trained again.
        # This is needed to compute the proposal prior for the next round.
        if resume and round_current == 0:

            if not os.path.exists(trained_model_path):
                raise FileNotFoundError(
                    "The folder specified in the config file at cfg['resume_training']['save_dir'] does not contain a trained_model.pkl file.\n"
                    "To use the resume mode, you need to specify the correct path."
                )

            with open(trained_model_path, "rb") as f:
                density_estimator = pickle.load(f)
            logger.info(
                f"Loaded pre-trained model for round {effective_round}, ensemble index {index}."
            )
        else:
            with timewith.TimeWith(
                f"[TrainingRound{effective_round}Ensemble{index}]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):
                train_args = {
                    "learning_rate": config["trainer"]["lr"],
                    "training_batch_size": config["trainer"]["batch_size"],
                    "validation_fraction": config["trainer"][
                        "validation_fraction"
                    ],
                    "show_train_summary": True,
                    "retrain_from_scratch": retrain_from_scratch,
                }
                # If model_type is 'snpe' and the proposal is directly the approximated posterior,
                # enable the correction for using a proposal distribution different from the prior.
                if (
                    model_type == "snpe"
                    and not config["trainer"]["truncated_prior"]
                ):
                    train_args["force_first_round_loss"] = True

                density_estimator = inference.append_simulations(
                    parameter_round.to(device),
                    matrix_round.to(device),
                    proposal=proposal,
                ).train(**train_args)

            logger.info(
                f"Trained density estimator for round {effective_round}, ensemble index {index}."
            )

            with open(trained_model_path, "wb") as output_file:
                pickle.dump(density_estimator.cpu(), output_file)
            logger.info(
                f"Saved trained model for round {effective_round}, ensemble index {index}."
            )

        if model_type == "snpe":
            posterior = inference.build_posterior(density_estimator.to(device))
        elif model_type == "snle":
            posterior = inference.build_posterior(
                density_estimator=density_estimator,
                mcmc_method=config["trainer"]["mcmc_sampler"],
                mcmc_parameters={"num_chains": 20, "thin": 5},
            )
        else:
            logger.exception(
                "The model type '{}' is not supported. ".format(model_type)
            )
            sys.exit(1)

        posteriors_list.append(posterior)

        logger.info(
            f"Saved inference for round {effective_round}, ensemble index {index}."
        )
        with open(inference_model_path, "wb") as inference_file:
            pickle.dump(inference, inference_file)

        # Saving the training statistics.
        save_training_statistics(config, inference, index, effective_round)

    if ensemble:
        # Giving each network in the ensemble an equal weight.
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list, weights=weights_ensemble.to(device)
        )
    else:
        final_posterior = posteriors_list[0]

    return final_posterior


def initialize_prior(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    dataset: dl.DatasetMultichannelArray,
) -> BoxUniform:
    """
    Initializes the prior distribution for the model based on the configuration and dataset.

    The prior is initialized as a uniform distribution over the parameter space.
    If the dataset is normalized or standardized, the prior is scaled accordingly to ensure
    that it fits the transformed space. If no normalization or standardization is applied,
    the prior is defined over the raw parameter space as specified in the configuration.

    Args:
        config (configuration_parser.ConfigurationParser): The configuration object containing the model settings.
        device (torch.device): The device on which the prior distribution is allocated (e.g., CPU or GPU).
        dataset (dl.DatasetMultichannelArray): The dataset where is saved the statistics for normalization or standardization.

    Returns:
        BoxUniform: The initialized prior distribution as a BoxUniform object.

    """
    n_parameters = len(torch.tensor(config["prior_ranges"]["low"]))
    # Setting the prior distribution for the parameters.
    # Note that we need to rescale the prior distribution to ensure that it has the correct limits when
    # restricted.
    if config["training_data_loader"]["normalize"]:
        # All the parameters are rescaled in the range [0, 1].
        prior = BoxUniform(
            low=torch.tensor(np.zeros(n_parameters)),
            high=torch.tensor(np.ones(n_parameters)),
            device=f"{device}",
        )
    elif config["training_data_loader"]["standardize"]:
        low = (
            torch.tensor(config["prior_ranges"]["low"]) - dataset.target_mean
        ) / dataset.target_std
        high = (
            torch.tensor(config["prior_ranges"]["high"]) - dataset.target_mean
        ) / dataset.target_std
        prior = BoxUniform(
            low=low,
            high=high,
            device=f"{device}",
        )
    else:
        # Set the prior range to the range of the parameters.
        prior = BoxUniform(
            low=torch.tensor(config["prior_ranges"]["low"]),
            high=torch.tensor(config["prior_ranges"]["high"]),
            device=f"{device}",
        )
    return prior


def load_posterior(
    config: configuration_parser.ConfigurationParser,
    logger: Logger,
    inference_list: Union[SNPE_C, List[SNPE_C]],
    device: torch.device,
    round_current: int,
) -> Union[DirectPosterior, NeuralPosteriorEnsemble]:
    """
    Load the trained density estimator for a given round.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        logger (Logger): Logger object.
        inference_list (Union[SNPE_C, List[SNPE_C]]): sbi inference object or list of inference objects for ensemble.
        device (torch.device): Device used for training.
        round_current (int): Current round number.

    Returns:
        (Union[DirectPosterior, NeuralPosteriorEnsemble]): Trained density estimator or ensemble of estimators.
    """
    ensemble = config["trainer"]["ensemble"]
    ensemble_size = config["trainer"]["size_ensemble"] if ensemble else 1
    model_type = config["trainer"]["type"]
    posteriors_list = []

    # Load_dir folder is where the trained_model.pkl is saved.
    load_dir = config["infer"]["load_dir"]
    save_dir_round = load_dir + f"/round_{round_current}"

    for index in range(ensemble_size):
        trained_model_path = (
            os.path.join(
                save_dir_round, f"trained_model_ensemble_{index}.pickle"
            )
            if ensemble
            else os.path.join(save_dir_round, "trained_model.pickle")
        )

        inference = inference_list[index if ensemble else 0]

        with open(trained_model_path, "rb") as f:
            density_estimator = pickle.load(f)
        logger.info(
            f"Loaded pre-trained model for round {round_current}, ensemble index {index}."
        )

        if model_type == "snpe":
            posterior = inference.build_posterior(density_estimator.to(device))
        elif model_type == "snle":
            posterior = inference.build_posterior(
                mcmc_method=config["trainer"]["mcmc_sampler"],
                mcmc_parameters={"num_chains": 20, "thin": 5},
            )
        else:
            logger.exception(
                "The model type '{}' is not supported. ".format(model_type)
            )
            sys.exit(1)

        posteriors_list.append(posterior)

    if ensemble:
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list, weights=weights_ensemble.to(device)
        )
    else:
        final_posterior = posteriors_list[0]

    return final_posterior
