"""
    Training script for truncated sequential neural likelihood estimation following Papamakarios et al. (2019).
    https://arxiv.org/abs/1805.07226).

    This script implements the sequential neural likelihood estimator using the sbi package.

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

    python train_snle.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import collections
import os
import pathlib
import pickle
import sys
import time
from logging import Logger
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.inference import SNLE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snle.snle_a import SNLE_A
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array as dl
import pypopsyn.learning.utils.sbi_utils as ut
import utilities.benchmark.timewith as timewith
from pypopsyn.learning.utils.request_device import request_device
from utilities.experiment_helpers.run_simulation_set_sbi import (
    initialize_dask_cluster,
)


def build_network_snle(
    device: torch.device,
    prior: utils.BoxUniform,
) -> SNLE_A:
    """
    Building the neural network (composed of the embedding net and the density estimator) using the configuration file
    specified in the arguments, and setting up the inference procedure.

    Args:.
        device (torch.device): Device used to run the script.
        prior (utils.BoxUniform): Prior distribution.

    Returns:
    """

    inference = SNLE(
        device=f"{device}",
        prior=prior,
    )

    return inference


def initialize_inference_snle(
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    prior: utils.BoxUniform,
    ensemble: bool = False,
) -> Union[List[SNLE_A], SNLE_A]:
    """
    Initialize inference objects using the provided configuration.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the neural network
            architecture and other settings.
        device (torch.device): Device used to run the script.
        prior (utils.BoxUniform): Prior distribution used in the inference process.
        ensemble (bool): Flag indicating if ensemble mode is enabled. Defaults to False.

    Returns:
        (Union[List[SNPE_C], SNPE_C]): A list of initialized inference objects.
    """
    inference_list = []

    # If ensemble is set to False, only one neural network will be used for training, resulting in a single inference
    # object. Otherwise, there will be as many inference objects as the number of components in the ensemble.
    for _ in range(config["trainer"]["size_ensemble"] if ensemble else 1):
        inference = build_network_snle(device, prior=prior)
        inference_list.append(inference)

    return inference_list


def prepare_dataset_sbi_snle(
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
    matrix = np.zeros((len(dataset), 32))

    # Load the pre-trained embedding network to encode each dataset sample into a latent vector.
    embedding_model_path = config["training_data_loader"]["trained_embedding"]
    with open(embedding_model_path, "rb") as f:
        emb_neural_net = pickle.load(f)

    for i, (x, theta) in enumerate(dataset):
        # Reshaping the matrix to have the channel number at the beginning.
        x = np.moveaxis(x, -1, 0)

        if list(x.shape) != input_shape:
            logger.error(
                "Mismatch between the shape of the input data x {} and the input shape specified "
                "in the configuration file {}".format(x.shape, input_shape)
            )
            sys.exit()

        x_embbeded = (
            emb_neural_net._embedding_net(torch.tensor(x)).detach().numpy()
        )
        if normalize:
            matrix[i] = (x_embbeded - x_embbeded.min()) / (
                x_embbeded.max() - x_embbeded.min()
            )
        else:
            matrix[i] = (x_embbeded - x_embbeded.mean()) / (
                x_embbeded.std() + 1e-8
            )

        parameter[i] = theta

    # Transforming the maps and labels into torch.tensors.
    parameter = torch.from_numpy(parameter).type(torch.float32)
    matrix = torch.from_numpy(matrix).type(torch.float32)
    return dataset, parameter, matrix


def posterior_snle(
    config: configuration_parser.ConfigurationParser,
    save_dir_round: pathlib.Path,
    logger: Logger,
    inference_list: Union[SNLE_A, List[SNLE_A]],
    parameter_round: torch.Tensor,
    matrix_round: torch.Tensor,
    device: torch.device,
    round_current: int,
    prof_log_path: str,
    prof_json_path: str,
    retrain_from_scratch: bool = False,
) -> Union[DirectPosterior, NeuralPosteriorEnsemble]:
    """
    Train the density estimator for a given round.

    If resuming is set to True, this mode allows training to continue from the last completed round if interrupted.
    It uses the previously saved state to resume training without starting over.
    If ensemble training is enabled, multiple models (an ensemble) are trained and their predictions are combined to
    ensure conservative coverages. Each of the neural networks will be trained on the same training dataset.

    Note that the inference object should be different for each component of the ensemble to ensure independent weights
    for each component.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        save_dir_round (pathlib.Path): Directory where the trained model will be saved or is saved already.
        logger (Logger): Logger object.
        inference_list (Union[SNPE_C, List[SNPE_C]]): sbi inference object or list of inference objects for ensemble.
        parameter_round (torch.Tensor): Tensor containing the parameters for the current round.
        matrix_round (torch.Tensor): Tensor containing the matrices for the current round.
        device (torch.device): Device used for training.
        round_current (int): Current round number.
        prof_json_path (str): The profile.json path.
        prof_log_path (str): The profile.log path.
        retrain_from_scratch (bool): Whether to retrain the conditional density estimator for the posterior from
            scratch each round. Default value is False.

    Returns:
        (Union[DirectPosterior, NeuralPosteriorEnsemble]): Trained density estimator or ensemble of estimators.
    """
    ensemble = config["trainer"]["ensemble"]
    ensemble_size = config["trainer"]["size_ensemble"] if ensemble else 1
    resume = config["resume_training"]["resume"]
    last_round = config["resume_training"]["last_round"]

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
                density_estimator = inference.append_simulations(
                    parameter_round.to(device), matrix_round.to(device)
                ).train(
                    learning_rate=config["trainer"]["lr"],
                    training_batch_size=config["trainer"]["batch_size"],
                    validation_fraction=config["trainer"][
                        "validation_fraction"
                    ],
                    show_train_summary=True,
                    retrain_from_scratch=retrain_from_scratch,
                )
            logger.info(
                f"Trained density estimator for round {effective_round}, ensemble index {index}."
            )

            with open(trained_model_path, "wb") as output_file:
                pickle.dump(density_estimator.cpu(), output_file)
            logger.info(
                f"Saved trained model for round {effective_round}, ensemble index {index}."
            )

        with open(inference_model_path, "wb") as output_file:
            pickle.dump(inference.cpu(), output_file)
        logger.info(
            f"Saved inference object for round {effective_round}, ensemble index {index}."
        )
        posterior = inference.build_posterior(
            mcmc_method=config["trainer"]["mcmc_sampler"],
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        posteriors_list.append(posterior)

        if not retrain_from_scratch:
            logger.info(
                f"Saved inference for round {effective_round}, ensemble index {index}."
            )
            with open(inference_model_path, "wb") as inference_file:
                pickle.dump(inference, inference_file)

        # Saving the training statistics.
        ut.save_training_statistics(config, inference, index, effective_round)

    if ensemble:
        # Giving each network in the ensemble an equal weight.
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list, weights=weights_ensemble.to(device)
        )
    else:
        final_posterior = posteriors_list[0]

    return final_posterior


def train(
    args: argparse.Namespace, config: configuration_parser.ConfigurationParser
) -> None:
    """
    Training a density estimator to infer the posterior distribution at the observed population with the truncated
    sequential neural posterior estimator approach in Deistler et al. (2022) using the sbi package.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse. It includes:

            - configuration (str): Path to the configuration file.
            - plot_proposal (bool): If set to True, generates proposal corner plots for each round.
            - trained_model (str): Path to the pretrained model (this argument is not used here).
            - infer (str): Flag to set up the inference saving path (default is True).

        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.
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
    resume = config["resume_training"]["resume"]
    ensemble = config["trainer"]["ensemble"]
    retrain_from_scratch = config["trainer"]["retrain_from_scratch"]

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

            # If resuming from a previous training run, first create the training dataset for the first round
            # by merging all the training datasets from the previously completed rounds.

            if resume:
                last_completed_round = int(
                    config["resume_training"]["last_round"]
                )
                train_dataset_all_round_path = pathlib.Path().joinpath(
                    config["training_data_loader"]["dataset_path"],
                    "generated_dataset",
                )
                train_dataset_path = ut.merge_all_rounds_dataset(
                    train_dataset_all_round_path, last_completed_round
                )
            else:
                train_dataset_path = config["training_data_loader"][
                    "dataset_path_first_round"
                ]
            logger.info(
                "Preparing the training dataset for sbi for the first round..."
            )
            dataset, parameter, matrix = prepare_dataset_sbi_snle(
                train_dataset_path, config, logger
            )
            n_parameters = len(torch.tensor(config["prior_ranges"]["low"]))
            num_rounds = config["trainer"]["num_rounds"]

            # Loading the train dataset as a dataframe and extracting the ground truth labels.
            filter_labels = config["training_data_loader"]["filter_labels"]
            dataset_df = pd.read_csv(train_dataset_path + "/dataset_full.csv")
            parameter_labels = dataset_df.columns[filter_labels]

            if config["set_manual_seed"] is True:
                torch.manual_seed(config["manual_seed"])
                logger.info("Seed: {}".format(config["manual_seed"]))
            else:
                torch.manual_seed(int(time.time()))
                logger.info("Seed: {}".format(int(time.time())))

            logger.info("Defining the prior distribution...")

            # Setting the prior distribution for the parameters.
            # Note that we need to rescale the prior distribution to ensure that it has the correct limits when
            # restricted.
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

            logger.info("Building the neural network...")

            # When resuming from a previous run, load the inference object that contains the weights of the previously
            # trained neural networks. Otherwise, initialize the neural network. Note that if resume = True and
            # retrain_from_scratch = True, in the first round we load the trained model. Hence, there is no need to load
            # the inference object as all the information about the weights is already in the trained model.

            if resume and not retrain_from_scratch:
                inference_list = ut.load_inference(
                    config, last_completed_round, ensemble
                )
            else:
                inference_list = initialize_inference_snle(
                    config, device, prior, ensemble
                )

            # Create the matrix for the observed sample of neutron stars.
            _, _, x_o = prepare_dataset_sbi_snle(
                config["observed_sample"]["dataset_path"],
                config,
                logger,
                atnf=True,
            )
            proposal = prior

            # Lists to store parameters and matrices from each round.
            parameter_list = []
            matrix_list = []

        for i in range(num_rounds):

            # Creating a folder to save the model, coverage and posterior distribution for each round.
            # If resuming from a previous run, compute the effective round number to continue from.
            if resume:
                effective_round = i + int(
                    config["resume_training"]["last_round"]
                )
                save_dir_round = pathlib.Path(
                    config["resume_training"]["save_dir"]
                ) / pathlib.Path(f"round_{effective_round}")

            else:
                effective_round = i
                save_dir_round = config.save_dir / f"round_{i}"
            save_dir_round.mkdir(parents=True, exist_ok=True)

            with timewith.TimeWith(
                f"[TotalRound{effective_round}]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):
                with timewith.TimeWith(
                    f"[TrainingRound{effective_round}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):

                    num_sim_train = config["training_data_loader"]["num_sim"]

                    # In the first round, instead of simulating the training dataset, we use the simulations
                    # previously run.
                    if i > 0:
                        logger.info(
                            f"Training, ------------------------------- round {effective_round} -------------------------------------"
                        )

                        logger.info(
                            "Simulating the training dataset for {} simulations...".format(
                                num_sim_train
                            )
                        )
                        train_dataset_path = ut.wrapper_pypopsyn(
                            proposal,
                            config=config,
                            num_sim=num_sim_train,
                            effective_round=effective_round,
                            test=False,
                            dataset=dataset,
                            device=device,
                        )

                        # Building the training dataset for sbi.
                        logger.info(
                            "Preparing the training data set for sbi..."
                        )
                        dataset, parameter, matrix = prepare_dataset_sbi_snle(
                            train_dataset_path, config, logger
                        )

                    # Saving the training data to reuse it in the next rounds.
                    parameter_list.append(parameter)
                    matrix_list.append(matrix)
                    parameter_round = torch.cat(parameter_list, dim=0)
                    matrix_round = torch.cat(matrix_list, dim=0)

                    logger.info(
                        f"Training the density estimator with {parameter_round.shape[0]} samples in round {effective_round} ..."
                    )

                    posterior = posterior_snle(
                        config=config,
                        save_dir_round=save_dir_round,
                        logger=logger,
                        inference_list=inference_list,
                        parameter_round=parameter_round,
                        matrix_round=matrix_round,
                        device=device,
                        round_current=i,
                        prof_log_path=prof_log_path,
                        prof_json_path=prof_json_path,
                        retrain_from_scratch=retrain_from_scratch,
                    )

                # If compute_coverage is equal to True, simulate a test dataset and compute its coverage probability.
                if config["trainer"]["compute_coverage"]:
                    with timewith.TimeWith(
                        f"[TestingRound{i}]",
                        prof_log_path,
                        prof_json_path,
                        config["show_profiling"],
                    ):

                        num_sim_test = config["test_data_loader"]["num_sim"]

                        if i == 0:
                            logger.info(
                                f"Loading the test dataset for round {effective_round}..."
                            )

                            # If resuming from a previous training run, we do not create the test dataset in the first
                            # iteration. Instead, we use the test dataset from the last completed round.
                            if resume:
                                test_dataset_path = str(
                                    pathlib.Path().joinpath(
                                        config["test_data_loader"][
                                            "dataset_path"
                                        ],
                                        f"generated_dataset/round_{effective_round}",
                                    )
                                )

                            else:
                                test_dataset_path = config["test_data_loader"][
                                    "dataset_path_first_round"
                                ]

                        else:
                            logger.info(
                                "Simulating the test dataset for {} simulations...".format(
                                    num_sim_test
                                )
                            )
                            test_dataset_path = ut.wrapper_pypopsyn(
                                proposal,
                                config=config,
                                num_sim=num_sim_test,
                                effective_round=effective_round,
                                test=True,
                                dataset=dataset,
                                device=device,
                            )
                        (
                            _,
                            parameter_test,
                            matrix_test,
                        ) = ut.prepare_dataset_sbi(
                            test_dataset_path, config, logger
                        )
                        logger.info(
                            f"Computing the ranks and the coverage probability for round_{effective_round}"
                        )
                        ut.compute_rank_coverage(
                            save_dir=save_dir_round,
                            parameter=parameter_test,
                            matrix=matrix_test,
                            posterior=posterior,
                            device=device,
                            parameter_labels=parameter_labels,
                            logger=logger,
                            effective_round=effective_round,
                        )

                with timewith.TimeWith(
                    f"[ComputeRestrictedPriorRound{effective_round}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    logger.info(
                        f"Computing the proposal prior for round {effective_round + 1}..."
                    )

                    posterior_obs = posterior.set_default_x(x_o)
                    proposal = posterior_obs

                logger.info(
                    f"Inferring the parameters for the observed sample for round {effective_round}..."
                )

                observed_samples_posterior = posterior_obs.sample(
                    (10000,), show_progress_bars=True
                ).cpu()
                ut.corner_plot(
                    observed_samples_posterior,
                    dataset,
                    f"{save_dir_round}/corner_plot_observed_sample_{effective_round}.pdf",
                )
                torch.save(
                    observed_samples_posterior,
                    f"{save_dir_round}/samples_posterior_{effective_round}.pt",
                )

                # If retrain_from_scratch is set to True, initialize the inference object to reset the weights
                # and avoid reusing the previously trained weights at each round.

                if retrain_from_scratch:
                    inference_list = initialize_inference_snle(
                        config, device, prior, ensemble
                    )

            # Stop the training when the number of rounds is reached. This is necessary in the resume case to avoid
            # performing extra rounds, since the iteration counter (i) does not reflect the effective round number.
            if effective_round == num_rounds - 1:
                break

        if config["enable_dask"]:
            # Closing the cluster once the training has finished.
            cluster.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Truncated SNPE trainer")

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="pypopsyn/learning/config_tsnpe.json",
        help="Machine learning configuration file path.",
    )

    args.add_argument(
        "--plot_proposal",
        type=configuration_parser.str_to_bool,
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
            type=configuration_parser.str_to_bool,
            nargs="?",
            target="training_data_loader;normalize",
        ),
        CustomArgs(
            ["--standardize"],
            type=configuration_parser.str_to_bool,
            nargs="?",
            target="training_data_loader;standardize",
        ),
        CustomArgs(["--lr"], type=float, nargs="?", target="trainer;lr"),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(args)

    train(args.parse_args(), configuration)
