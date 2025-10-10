"""
    Inference script for simulation-based inference.

    This script performs inference on a test dataset within a SNPE or SNLE framework using the sbi package. It loads the
    trained density estimator to approximate the posterior distribution for a dataset of simulated data and evaluates
    its performance on a test dataset in each round.

    Display help message to run the code:

    python sbi_infer.py --help

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


def infer(config: configuration_parser.ConfigurationParser) -> None:
    """
    Infer the posterior distribution for the observed population using simulation-based inference, assuming that the
    sbi_train.py script has already been run and a trained_model.pkl file has been generated.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying dataset loading parameters.
    """
    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("infer")
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
    ensemble = config["trainer"]["ensemble"]

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
            logger.info(
                "Loading the training dataset to extract the statistics..."
            )
            train_dataset_path = config["training_data_loader"][
                "dataset_path_round_0"
            ]

            # Load the training dataset information to extract the training statistics.
            dataset, _, _ = ut.prepare_dataset_sbi(
                train_dataset_path, config, logger
            )
            num_rounds = config["trainer"]["num_rounds"]

            # Loading the header of the training dataset to extract the ground truth labels.
            filter_labels = config["training_data_loader"]["filter_labels"]
            dataset_header = pd.read_csv(
                f"{train_dataset_path}/dataset_full.csv", nrows=0
            )
            parameter_labels = dataset_header.columns[filter_labels]

            if config["set_manual_seed"] is True:
                torch.manual_seed(config["manual_seed"])
                logger.info("Seed: {}".format(config["manual_seed"]))
            else:
                torch.manual_seed(int(time.time()))
                logger.info("Seed: {}".format(int(time.time())))

            logger.info("Defining the prior distribution...")

            # Setting the prior distribution for the parameters.
            prior = sbi_builder.initialize_prior(config, device, dataset)

            # Create the matrix for the observed sample of neutron stars.
            _, _, x_o = ut.prepare_dataset_sbi(
                config["observed_sample"]["dataset_path"],
                config,
                logger,
                atnf=True,
            )

        parameter_test = []
        matrix_test = []
        for i in range(num_rounds):

            save_dir_round = config.log_dir / f"round_{i}"
            save_dir_round.mkdir(parents=True, exist_ok=True)

            with timewith.TimeWith(
                f"[TotalRound{i}]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):

                if config["trainer"]["retrain_from_scratch"]:
                    inference_list = sbi_builder.initialize_inference(
                        config, device, prior, logger, ensemble
                    )
                else:
                    inference_list = sbi_builder.load_inference(
                        config, i, config["infer"]["load_dir"], ensemble
                    )

                final_posterior = sbi_builder.load_posterior(
                    config, logger, inference_list, device, i
                )
                posterior_obs = final_posterior.set_default_x(x_o)
                logger.info(
                    f"Sampling from the posterior for round {i + 1}..."
                )
                observed_samples_posterior = sample_without_nan(
                    posterior_obs,
                    sampling_size=5000,
                    device=device,
                ).cpu()

                # Setting the proposal prior to the truncated prior or to the previous approximated posterior
                # distribution at the observed data.
                if config["trainer"]["truncated_prior"]:
                    proposal = sbi_builder.compute_proposal_prior(
                        posterior_obs, config, prior, device
                    )

                else:
                    proposal = posterior_obs

                with timewith.TimeWith(
                    f"[ComputingCoverage{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    if config["infer"]["compute_coverage"]:
                        if config["infer"]["sim_dataset"] and i > 0:
                            num_sim_test = config["test_data_loader"][
                                "num_sim"
                            ]

                            logger.info(
                                "Simulating the test dataset for {} simulations...".format(
                                    num_sim_test
                                )
                            )

                            test_dataset_path = ut.wrapper_pypopsyn(
                                proposal,
                                config=config,
                                num_sim=num_sim_test,
                                effective_round=i,
                                test=True,
                                dataset=dataset,
                                device=device,
                            )
                        else:
                            test_dataset_path = config["test_data_loader"][
                                "dataset_path_round_0"
                            ]

                        (_, parameter, matrix,) = ut.prepare_dataset_sbi(
                            test_dataset_path, config, logger
                        )

                        # Saving the testing data to reuse it in the next rounds if the proposal is truncated with the
                        # prior.
                        if not config["trainer"]["truncated_prior"]:
                            parameter_test = []
                            matrix_test = []

                        parameter_test.append(parameter)
                        matrix_test.append(matrix)
                        parameter_round_test = torch.cat(parameter_test, dim=0)
                        matrix_round_test = torch.cat(matrix_test, dim=0)

                        logger.info(
                            f"Computing the ranks and the coverage probability for round_{i}"
                        )

                        ut.compute_rank_coverage(
                            save_dir=save_dir_round,
                            parameter=parameter_round_test,
                            matrix=matrix_round_test,
                            posterior=final_posterior,
                            device=device,
                            parameter_labels=parameter_labels,
                            logger=logger,
                            effective_round=i,
                        )

                ut.corner_plot(
                    observed_samples_posterior,
                    dataset,
                    f"{save_dir_round}/corner_plot_observed_sample.pdf",
                )
                torch.save(
                    observed_samples_posterior,
                    f"{save_dir_round}/samples_posterior.pt",
                )

        if config["enable_dask"]:
            # Closing the cluster once the training has finished.
            cluster.close()

        logger.info(f"Inference results have been saved to: {config.log_dir}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference SBI")

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
        default=True,
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

    infer(configuration)
