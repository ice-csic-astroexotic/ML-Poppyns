"""
    Training script for truncated sequential neural posterior estimation following Deistler et al. (2022).
    [https://arxiv.org/abs/2210.04815](https://arxiv.org/abs/2210.04815).

    This script implements the truncated sequential neural posterior estimator using the sbi package. It iteratively
    trains a density estimator for `num_rounds`, where each iteration involves generating training and testing datasets
    based on the previously approximated posterior distribution at the observed sample. This approach focuses on
    the region of the parameter space that matches the observed population to save computational resources.

    To create the training and test datasets, we use either the `multiprocessing` or `Dask`
    [https://www.dask.org/](https://www.dask.org/) package to run the simulations simultaneously in a multithreaded
    manner. To use Dask change the variable `enable_dask` in the configuration file to True. Otherwise, change it to
    False to use multiprocessing.

    Note that there is an option to resume training from a previous run. This allows for training over multiple rounds
    on a server. If the maximum wall time is reached or if any interruptions occur, the training can be resumed from
    the last completed round. To enable the resume mode, set the `resume_training` field to `True` in the configuration
    file. It is also necessary to specify where the logs and models were saved in the first run and indicate the last
    completed round.

    For further details, visit [https://www.mackelab.org/sbi/](https://www.mackelab.org/sbi/).

    Display help message to run the code:

    python train_tsnpe.py --help

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
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snle import SNLE_A
from sbi.inference.snpe import SNPE_C
from tqdm import tqdm

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array as dl
from pypopsyn.generator import generate_dataset_surveys
from utilities.coverage_probability import coverage_prob
from utilities.experiment_helpers.run_simulation_set_sbi import (
    simulator_dask,
    simulator_multiprocess,
)


def handler(signum: int, frame: FrameType) -> None:
    """
    Signal handler that raises a TimeoutError when a SIGALRM signal is received.

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
    n_samples_coverage: int,
) -> torch.Tensor:
    """
     Perform sampling from the posterior distribution. This function is designed to be used with a signal handler
     to enforce a timeout during sampling.

     Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples_coverage (int): The number of samples to draw from the posterior distribution.

    Returns:
        (torch.Tensor): The samples drawn from the posterior distribution.
    """
    return posterior.set_default_x(simulation_output).sample(
        (n_samples_coverage,), show_progress_bars=False
    )


def sample_with_timeout(
    posterior: DirectPosterior,
    simulation_output: torch.Tensor,
    n_samples_coverage: int,
    timeout: int = 180,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Perform sampling from the posterior distribution with a specified timeout.

    Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples_coverage (int): The number of samples to draw from the posterior distribution.
        timeout (int, optional): The maximum time in seconds to allow for n_samples_coverage to be drawn. Defaults to 180 seconds.

    Returns:
        (Tuple[Optional[torch.Tensor], bool]): A tuple containing the result of the sampling (or None if it times out)
            and a boolean indicating whether the sampling was successful.
    """
    # Set the signal handler and start the alarm to enforce the function timeout.
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        # If the sampling completes before the timeout, return the result and set success to True.
        result = sample(posterior, simulation_output, n_samples_coverage)
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
    n_samples_coverage: int,
    logger: Logger,
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
            posterior, simulation_output, n_samples_coverage, timeout=180
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
    ensemble: bool = False,
) -> Union[List[Union[SNPE_C, SNLE_A]], SNPE_C, SNLE_A]:
    """
    Load inference objects from pickle files. Note that this is used when resume mode is enabled.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the settings.
        round_number (int): The round number to load the inference from.
        ensemble (bool): Flag indicating if ensemble mode is enabled. Defaults to False.

    Returns:
        (Union[List[SNPE_C], SNPE_C]): A list of inference objects.
    """
    save_dir = config["resume_training"]["save_dir"]
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
                "The folder specified in the config file at cfg['resume_training']['save_dir'] does not contain a inference.pickle file.\n"
                "To use the resume mode, you need to specify the correct path."
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
    Simulating `num_sim` of mock neutron star population given the `proposal` distribution. After simulating the
    populations, we generate the compressed representations for the output.

    Args:
        proposal (Union[DirectPosterior,utils.RestrictedPrior]): Proposal distribution used for sampling the parameters.
        num_sim (int): Number of simulations to perform.
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        effective_round (int): Number of the effective round during the sequential inference approach.
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


def prepare_dataset_sbi(
    dataset_folder: str,
    config: configuration_parser.ConfigurationParser,
    logger: Logger,
    atnf: Optional[bool] = False,
) -> Tuple[dl.DatasetMultichannelArray, torch.tensor, torch.tensor]:
    """
    Prepare dataset for use in sbi training.

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
        inference (SNPE_C): sbi inference object.
        index (int): The ensemble index, if ensemble is set to False index is equal to 0.
        effective_round (int): Number of the effective round during the sequential inference approach.
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
    Compute the proposal prior by restricting the prior to the posterior of the observation.

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
    parameter_labels: list,
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
        effective_round (int): Number of the effective round during the sequential inference approach.
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
