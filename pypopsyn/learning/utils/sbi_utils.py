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
import sys
from logging import Logger
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
from sbi.inference.snle.snle_a import SNLE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from tqdm import tqdm

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array as dl
import pypopsyn.learning.utils.sampler as sampler
from pypopsyn.generator import generate_dataset_surveys
from utilities.coverage_probability import coverage_prob
from utilities.experiment_helpers.run_simulation_set_sbi import (
    simulator_dask,
    simulator_multiprocess,
)


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
        n_samples (int): The number of samples to draw from the posterior distribution.
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
        posterior_samples, success = sampler.sample_with_timeout(
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
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
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
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
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
        save_dir (pathlib): Directory to save the computed results and plots.
        parameter (torch.Tensor): Tensor containing the parameters for the test dataset in the current round.
        matrix (torch.Tensor): Tensor containing the matrices for the test dataset in the current round.
        posterior (DirectPosterior): Approximated posterior distribution.
        device (torch.device): Device used to run the script.
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
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        logger (Logger): Logger object.
        atnf (bool, optional): Indicates whether the PPdot density maps in the 'train_data_set' folder correspond to
            the observed population or to a simulated population. If set to True, the simulations correspond to the
            observed ATNF population. The default is False.

    Returns:
        (tuple): A tuple containing the dataset containing the statistics, parameter tensor and input matrix tensor.
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
            if standardize:
                matrix[i] = (x_embedded - x_embedded.mean()) / (
                    x_embedded.std() + 1e-8
                )
            else:
                matrix[i] = x
        else:
            matrix[i] = x

        parameter[i] = theta

    # Transforming the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)

    return dataset, parameter, matrix
