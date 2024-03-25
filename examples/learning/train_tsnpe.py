import argparse
import collections
import json
import pathlib
import pickle
import subprocess
import sys
import time
from typing import Optional, Tuple

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi import utils
from sbi.inference import SNPE
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.snpe.snpe_c import SNPE_C
from torch.distributions import Distribution
from tqdm import tqdm

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device
from pypopsyn.simulator.configuration import cfg
from scripts.coverage_probability import coverage_prob
from scripts.simulation_helper_sbi import simulator

_ = torch.manual_seed(0)


def calculate_smallest_hdr(
    posterior: NeuralPosterior,
    theta: torch.tensor,
    matrix: torch.tensor,
    n_samples_coverage: int,
    device: Optional[torch.device] = "cpu",
) -> np.ndarray:
    """
    Calculating the smallest highest density region of the posterior, that contains the true value for the test
    dataset in `theta` and `matrix`.

    Args:
        posterior (NeuralPosterior): Posterior distribution.
        theta (torch.tensor): Tensor containing the values of the parameters used to generate the simulated
         population in matrix.
        matrix (torch.tensor): Tensor containing the maps of the simulated population.
        n_samples_coverage (float): Number of approximate posterior samples used for computing the coverage.
        device (Optional[torch.device]): Device used to run the script. Defaults to 'cpu'.

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


def import_statistics(stats_path: str) -> Tuple[torch.tensor, torch.tensor]:
    """
    Extracting the mean and standard deviation for all the parameters in the `stats_path` file.
    Args:
        stats_path (str): Path to the file where the statistics are saved.
    Returns:
        (torch.tensor, torch.tensor): Mean and standard deviation for the parameters in the `stats_path` file.
    """
    std_list = []
    mean_list = []
    with open(stats_path, "r") as json_file:
        data = json.load(json_file)
    for key, value in data.items():
        std_list.append(value["std"])
        mean_list.append(value["mean"])
    mean = torch.tensor(mean_list)
    std = torch.tensor(std_list)
    return mean, std


def build_network(
    config: configuration_parser.ConfigurationParser, device: torch.device
) -> SNPE_C:
    """
    Building the neural network using the configuration file specify in the arguments.
    Args:
        config: Configuration object specifying the neural network architecture and other settings.
        device (torch.device): device in which the model will be executed.
    Returns:
        inference (sbi.inference.snpe.snpe_c.SNPE_C): An instance of sbi snpe inference object.
    """

    embedding_net = config.init_object("arch", learning_models)

    # Initialize weights ---------------------------------------------------

    weight_initializer = config.init_object(
        "weights_initializer", learning_initializers
    )
    # Apply the weight initialization scheme to every layer in the model.
    embedding_net.apply(weight_initializer)
    hidden_features = config["arch"]["args"]["len_output_layer"]

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

    # Set up the inference procedure -----------------------------
    # By default the procedure is the SNPE-C (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
    inference = SNPE(
        density_estimator=neural_posterior,
        device=f"{device}",
    )

    return inference


def wrapper_pypopsyn(
    proposal: Distribution,
    num_sim: int,
    config: configuration_parser.ConfigurationParser,
    nround: int,
    test: bool,
    dataset: dl.DatasetMultichannelArray,
) -> str:
    """
    Simulating `num_sim` of mock neutron star population given the `proposal` distribution.

    Args:
        proposal (Distribution): Proposal distribution used for sampling the parameters.
        num_sim (int): Number of simulations to perform.
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        nround (int): Round number.
        test (bool): Flag indicating whether the simulations are for testing or training.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.
    Returns:
        str: Path to the generated dataset.
    """
    if test:
        sim_dir_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/simulations/round_{nround}"
        )
        dataset_path = (
            config["test_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{nround}"
        )
    else:
        sim_dir_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/simulations/round_{nround}"
        )
        dataset_path = (
            config["training_data_loader"]["dataset_path"]
            + f"/generated_dataset/round_{nround}"
        )

    dyn_data_path = config["dyn_data_loader"]["dataset_path"]
    args_simulator = argparse.Namespace(
        dyn_data=dyn_data_path,
        output_dir=sim_dir_path,
        simulator_type="simulate_population_magrot_det",
        sampling_type="prior",
        sampling_size=num_sim,
        P_initial_log10_mean=[
            config["prior_ranges"]["low"][2],
            config["prior_ranges"]["high"][2],
        ],
        P_initial_log10_sigma=[
            config["prior_ranges"]["low"][3],
            config["prior_ranges"]["high"][3],
        ],
        B_initial_log10_mean=[
            config["prior_ranges"]["low"][0],
            config["prior_ranges"]["high"][0],
        ],
        B_initial_log10_sigma=[
            config["prior_ranges"]["low"][1],
            config["prior_ranges"]["high"][1],
        ],
        a_late=[
            config["prior_ranges"]["low"][4],
            config["prior_ranges"]["high"][4],
        ],
        processes=config["n_processes"],
        kick_model="km_maxwell",
        sigma_k=None,
        h_c=None,
        spin_period_model="log-normal",
    )

    server_path = cfg["path_server_software"]
    command = [
        "python",
        f"{server_path}examples/generator/generate_dataset_surveys.py",
        "--data",
        str(sim_dir_path),
        "--save_dir",
        str(dataset_path),
        "--resolution_ppdot",
        str(config["arch"]["args"]["input_shape"][1]),
        "--data_type",
        "array",
    ]

    simulator(args_simulator, proposal, dataset)
    subprocess.run(command)

    return dataset_path


def corner_plot(
    posterior: NeuralPosterior,
    dataset: dl.DatasetMultichannelArray,
    save_dir: str,
) -> None:
    """
    Plotting the corner plot for the posterior distribution.

    Args:
        posterior (NeuralPosterior): Posterior distribution for performing inference.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.
        save_dir (str): Directory to save the corner plot.

    Returns:
        None
    """
    # Save the corner plot for the observed sample
    observed_samples = posterior.sample(
        (50000,), show_progress_bars=False
    ).cpu()

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

    # Save the samples from the inferred posterior distribution.
    torch.save(
        observed_samples,
        f"{save_dir}/samples.pt",
    )

    # Saving the best estimated parameters and the 95% CI into the log.txt file.
    quantile = np.quantile(observed_samples, [0.025, 0.5, 0.975], axis=0)

    range_param = [[par_min[v], par_max[v]] for v in range(len(par_max))]

    param_median = quantile[1, :]

    # Corner plot of the inferred posterior distributions for each parameter.
    figure = corner.corner(
        observed_samples.detach().cpu().numpy(),
        bins=32,
        labels=["B_mu", "B_sigma", "P_mu", "P_sigma", "a_late"],
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
    atnf: Optional[bool] = False,
) -> Tuple[dl.DatasetMultichannelArray, torch.tensor, torch.tensor]:
    """
    Prepare dataset for use in SBI training.

    Args:
        train_data_set (str): Path to the training dataset.
        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.
        atnf (bool, optional): Whether to use ATNF dataset. Defaults to False.

    Returns:
        tuple: A tuple containing the dataset, parameter tensor and input matrix tensor.
    """

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

    # Load the training dataset ----------------------------------------------------------
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
        sys.exit(1)

    parameter = np.zeros((len(dataset), n_parameters))
    matrix = np.zeros(
        (len(dataset), input_shape[0], input_shape[1], input_shape[2])
    )
    for i, (x, theta) in enumerate(dataset):
        # Re-shape the matrix to have the channel number at the beginning.
        x = np.moveaxis(x, -1, 0)

        if list(x.shape) != input_shape:
            sys.exit()

        matrix[i] = x
        parameter[i] = theta

    # Transform the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)
    return dataset, parameter, matrix


def train(config):
    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("train")
    logger.info("Logger initialized...")

    # Set up GPU device if available.
    device, device_ids = request_device(config["n_gpu"])

    logger.info("Defining the prior distribution...")

    # Loading the statistics to apply the rescaling to the prior distribution.
    stats_path = config["training_data_loader"]["statistic_path"]
    mean, std = import_statistics(stats_path)

    n_parameters = len(torch.tensor(config["prior_ranges"]["low"]))
    num_rounds = config["training_data_loader"]["n_rounds"]

    if config["set_manual_seed"] is True:
        torch.manual_seed(config["manual_seed"])
        logger.info("Seed: {}".format(config["manual_seed"]))
    else:
        torch.manual_seed(int(time.time()))
        logger.info("Seed: {}".format(int(time.time())))

    # Set prior distribution for the parameters ------------------------------------------
    if config["training_data_loader"]["normalize"]:
        # All the parameters are rescaled in the range [0, 1].
        prior = utils.BoxUniform(
            low=torch.tensor(np.zeros(n_parameters)),
            high=torch.tensor(np.ones(n_parameters)),
            device=f"{device}",
        )
    elif config["training_data_loader"]["standardize"]:
        low = (
            torch.tensor(config["prior_ranges"]["low"]) - mean[0:n_parameters]
        ) / std[0:n_parameters]
        high = (
            torch.tensor(config["prior_ranges"]["high"]) - mean[0:n_parameters]
        ) / std[0:n_parameters]
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

    # At the first round we set the proposal equal to the prior.
    proposal = prior

    # Lists to store parameters and matrices from each round.
    parameter_list = []
    matrix_list = []

    logger.info("Building the neural network...")
    inference = build_network(config, device)

    # Initialize dataset with a default value to avoid warning in the first round.
    dataset = None

    for i in range(num_rounds):

        logger.info(
            f"Training, ------------------------------- round {i}-------------------------------------"
        )
        num_sim_train = config["training_data_loader"]["n_sim_round"]

        # If it's the first round, instead of simulating the training dataset, we use the simulation previously run.
        if i == 0:
            logger.info(
                "Loading the training dataset for for the first round..."
            )
            train_dataset_path = config["training_data_loader"][
                "dataset_path_first_round"
            ]
        else:
            logger.info(
                "Simulating the training dataset for {} simulations...".format(
                    num_sim_train
                )
            )
            train_dataset_path = wrapper_pypopsyn(
                proposal,
                config=config,
                num_sim=num_sim_train,
                nround=i,
                test=False,
                dataset=dataset,
            )

        # Building the training dataset for sbi.
        logger.info("Preparing the training data set for sbi...")
        dataset, parameter, matrix = prepare_dataset_sbi(
            train_dataset_path, config
        )

        # TODO: take care of the rescaling of the parameters for different rounds.
        # Saving the training data to reuse it in the next rounds.
        parameter_list.append(parameter)
        matrix_list.append(matrix)
        parameter_round = torch.cat(parameter_list, dim=0)
        matrix_round = torch.cat(matrix_list, dim=0)

        logger.info(
            f"Training the density estimator with {parameter_round.shape[0]} samples in round {i} ..."
        )
        density_estimator = inference.append_simulations(
            parameter_round.to(device), matrix_round.to(device)
        ).train(
            learning_rate=config["trainer"]["lr"],
            training_batch_size=config["trainer"]["batch_size"],
            validation_fraction=config["trainer"]["validation_fraction"],
            show_train_summary=True,
            force_first_round_loss=True,
        )

        posterior = inference.build_posterior(
            density_estimator, prior=proposal
        )

        num_sim_test = config["test_data_loader"]["n_sim_round"]

        if i == 0:
            logger.info("Loading the test dataset for the first round...")
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
                nround=i,
                test=True,
                dataset=dataset,
            )

            # Building the test dataset for sbi.
        _, parameter_test, matrix_test = prepare_dataset_sbi(
            test_dataset_path, config
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
        coverage_save_dir = pathlib.Path().joinpath(
            config.save_dir, f"/coverage_round{i}.pdf"
        )
        coverage_prob(hdr, n_betas=12, save_dir=coverage_save_dir)

        logger.info(f"Computing the proposal prior for round {i+1}...")
        # Create the matrix for the observed sample of neutron stars.
        _, _, x_o = prepare_dataset_sbi(
            config["observed_sample"]["dataset_path"], config, atnf=True
        )
        # Here we set the density of the posterior that we want to take to then restricted our prior.
        posterior_obs = posterior.set_default_x(x_o)
        accept_reject_fn = utils.get_density_thresholder(
            posterior_obs, quantile=1e-4, num_samples_to_estimate_support=10000
        )
        # -------Computing the proposal by using the restricted prior to the posterior at the observation.----------
        # If I use rejection method to compute the restricted proposal from the prior.
        # proposal = utils.RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
        # If I use SIR method to compute the restricted proposal from the prior.
        proposal = utils.RestrictedPrior(
            prior, accept_reject_fn, posterior=posterior_obs, sample_with="sir"
        )
        corner_plot(
            proposal,
            dataset,
            f"{config.save_dir}/corner_plot_prior_round_{i}.pdf",
        )

        logger.info(f"Saving the trained model for round {i}...")

        with open(
            f"{config.save_dir}/trained_model_{i}.pickle", "wb"
        ) as output_file:
            pickle.dump(density_estimator.cpu(), output_file)

        logger.info(
            f"Inferring the parameters for the observed sample for round {i}..."
        )
        corner_plot(
            posterior_obs,
            dataset,
            f"{config.save_dir}/corner_plot_observed_sample_{i}.pdf",
        )


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

    configuration = configuration_parser.ConfigurationParser.from_args(args)

    train(configuration)
