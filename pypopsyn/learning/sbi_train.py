"""
    Training script for simulation-based inference using the sbi package.

    It offers the flexibility to train either a Sequential Neural Posterior Estimator (SNPE) or a Sequential Neural
    Likelihood Estimator (SNLE), for the latter sampling from the posterior involves a MCMC sampler.
    This script implements a sequential approach which iteratively trains a density estimator for `num_rounds`,
    where each iteration involves generating training and testing datasets based on the previously approximated posterior
    distribution at the observed sample. This approach focuses on the region of the parameter space that matches
    the observed population to save computational resources.

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

    python sbi_train.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import collections
import pathlib
import time

import pandas as pd
import torch

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.utils.sbi_builder as sbi_builder
import pypopsyn.learning.utils.sbi_utils as ut
import utilities.benchmark.timewith as timewith
from pypopsyn.learning.utils.request_device import request_device
from utilities.experiment_helpers.run_simulation_set_sbi import (
    initialize_dask_cluster,
    sample_without_nan,
)


def train(config: configuration_parser.ConfigurationParser) -> None:
    """
    Training a density estimator to infer the posterior distribution at the observed population.

    Args:
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
            dataset, parameter, matrix = ut.prepare_dataset_sbi(
                train_dataset_path, config, logger
            )
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
            prior = sbi_builder.initialize_prior(config, device, dataset)
            logger.info("Building the neural network...")

            # When resuming from a previous run, load the inference object that contains the weights of the previously
            # trained neural networks. Otherwise, initialize the neural network. Note that if resume = True and
            # retrain_from_scratch = True, in the first round, we will train with newly initialized weights. Hence,
            # there is no need to load the inference object. To perform inference, the trained_model.pickle is the only
            # object needed.
            if resume and not retrain_from_scratch:
                inference_list = sbi_builder.load_inference(
                    config,
                    last_completed_round,
                    config["resume_training"]["save_dir"],
                    ensemble,
                )
            else:

                inference_list = sbi_builder.initialize_inference(
                    config, device, prior, logger, ensemble
                )

            # Create the matrix for the observed sample of neutron stars.
            _, _, x_o = ut.prepare_dataset_sbi(
                config["observed_sample"]["dataset_path"],
                config,
                logger,
                atnf=True,
            )

            # Set the proposal prior to the pior in the first round.
            proposal = prior

            # Lists to store parameters and matrices from each round.
            parameter_train = []
            matrix_train = []
            parameter_test = []
            matrix_test = []

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
                        dataset, parameter, matrix = ut.prepare_dataset_sbi(
                            train_dataset_path, config, logger
                        )

                    # Save the training and testing data for reuse in future rounds if the proposal distribution
                    # config["trainer"]["append_simulations"] is True.
                    # Otherwise, use the simulations from the current round.
                    if not config["trainer"]["append_simulations"]:
                        parameter_train = []
                        matrix_train = []
                        parameter_test = []
                        matrix_test = []

                    parameter_train.append(parameter)
                    matrix_train.append(matrix)
                    parameter_round = torch.cat(parameter_train, dim=0)
                    matrix_round = torch.cat(matrix_train, dim=0)

                    logger.info(
                        f"Training the density estimator with {parameter_round.shape[0]} samples in round {effective_round} ..."
                    )

                    # If the model is trained using an SNPE approach and the proposal prior is set to the approximated
                    # posterior, we need to correct the loss function accordingly when training with this new proposal
                    # prior.

                    if (
                        config["trainer"]["type"] == "snpe"
                        and not config["trainer"]["truncated_prior"]
                    ):
                        proposal_train = proposal
                    else:
                        proposal_train = None

                    posterior = sbi_builder.train_posterior(
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
                        proposal=proposal_train,
                    )

                if config["test_data_loader"]["testing"]:
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
                        (_, parameter, matrix,) = ut.prepare_dataset_sbi(
                            test_dataset_path, config, logger
                        )

                        parameter_test.append(parameter)
                        matrix_test.append(matrix)
                        parameter_round_test = torch.cat(parameter_test, dim=0)
                        matrix_round_test = torch.cat(matrix_test, dim=0)

                        logger.info(
                            f"Computing the ranks and the coverage probability for round_{effective_round}"
                        )
                        ut.compute_rank_coverage(
                            save_dir=save_dir_round,
                            parameter=parameter_round_test,
                            matrix=matrix_round_test,
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

                    if config["trainer"]["truncated_prior"]:
                        proposal = sbi_builder.compute_proposal_prior(
                            posterior_obs, config, prior, device
                        )

                        if config["trainer"]["plot_proposal"]:
                            # If config['trainer']['plot_proposal'] is set to True, a corner plot of the proposal
                            # distribution will be produced. Note that this might take a while  in the case of TSNPE
                            # since we are using SIR or rejection methods to sample from the proposal distribution.
                            observed_samples_proposal = sample_without_nan(
                                posterior_obs,
                                sampling_size=5000,
                                device=device,
                            ).cpu()
                            ut.corner_plot(
                                observed_samples_proposal,
                                dataset,
                                f"{save_dir_round}/corner_plot_prior_round_{effective_round + 1}.pdf",
                            )
                            # Save the samples from the inferred posterior distribution.
                            torch.save(
                                observed_samples_proposal,
                                f"{save_dir_round}/samples_prior_{effective_round + 1}.pt",
                            )
                    else:
                        proposal = posterior_obs

                logger.info(
                    f"Inferring the parameters for the observed sample for round {effective_round}..."
                )

                observed_samples_posterior = sample_without_nan(
                    posterior_obs, 50000, device
                ).cpu()
                ut.corner_plot(
                    observed_samples_posterior,
                    dataset,
                    f"{save_dir_round}/corner_plot_observed_sample.pdf",
                )
                torch.save(
                    observed_samples_posterior,
                    f"{save_dir_round}/samples_posterior.pt",
                )

                # If retrain_from_scratch is set to True, initialize the inference object to reset the weights
                # and avoid reusing the previously trained weights at each round.

                if retrain_from_scratch:
                    inference_list = sbi_builder.initialize_inference(
                        config, device, prior, logger, ensemble
                    )

            # Stop the training when the number of rounds is reached. This is necessary in the resume case to avoid
            # performing extra rounds, since the iteration counter (i) does not reflect the effective round number.
            if effective_round == num_rounds - 1:
                break

        if config["enable_dask"]:
            # Closing the cluster once the training has finished.
            cluster.close()

        logger.info(f"Training results have been saved to: {config.save_dir}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="SBI trainer")

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="pypopsyn/learning/config_sbi.json",
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

    train(configuration)
