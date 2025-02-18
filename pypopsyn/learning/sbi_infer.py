"""
    Inference script for sbi.

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
import sys
import time

import pandas as pd
import torch
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.utils.sbi_utils as ut
import utilities.benchmark.timewith as timewith
from pypopsyn.learning.utils.request_device import request_device


def infer(
    args: argparse.Namespace, config: configuration_parser.ConfigurationParser
) -> None:
    """
    Infer the posterior distribution for the observed population using the truncated sequential neural posterior
    estimator approach described in Deistler et al. (2022), assuming that the train_tsnpe.py script has already been run
    and a trained_model.pkl was generated.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse. It includes:

            - configuration (str): Path to the configuration file.
            - plot_proposal (bool): If set to True, generates proposal corner plots for each round.
            - trained_model (str): Path to the pretrained model (this argument is not used here).
            - infer (str): Flag to set up the inference saving path (default is True).

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
            logger.info(
                "Loading the training dataset to extract the statistics..."
            )
            train_dataset_path = config["training_data_loader"][
                "dataset_path_first_round"
            ]

            # Load the training dataset information. Note that when performing inference, we do not require the
            # underlying data samples, only the corresponding ground truths and their statistics.
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
            # Note that we need to rescale the prior distribution to ensure that it has the correct limits when
            # restricted.
            prior = ut.initialize_prior(config, device, dataset)

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

                model_type = config["trainer"]["type"]

                if model_type == "snle":
                    inference_list = ut.load_inference(
                        config, i, config["infer"]["load_dir"], ensemble
                    )
                    posteriors_list = []
                    for inference in inference_list:
                        posterior = inference.build_posterior(
                            mcmc_method=config["trainer"]["mcmc_sampler"],
                            mcmc_parameters={"num_chains": 20, "thin": 5},
                        )
                        posteriors_list.append(posterior)
                    if ensemble:
                        ensemble_size = len(inference_list)
                        # Giving each network in the ensemble an equal weight.
                        weights_ensemble = (
                            torch.ones(ensemble_size) / ensemble_size
                        )
                        final_posterior = NeuralPosteriorEnsemble(
                            posteriors_list,
                            weights=weights_ensemble.to(device),
                        )
                    else:
                        final_posterior = posteriors_list[0]

                elif model_type == "snpe":
                    inference_list = ut.initialize_inference(
                        config, device, prior, logger, ensemble
                    )
                    final_posterior = ut.load_posterior(
                        config, logger, inference_list, device, i
                    )
                else:

                    logger.exception(
                        "The model type '{}' is not supported. ".format(
                            model_type
                        )
                    )
                    sys.exit(1)

                posterior_obs = final_posterior.set_default_x(x_o)
                # Setting the proposal prior to the truncated prior or to the previous approximated posterior distribution at the observed data.
                if config["trainer"]["truncated_prior"]:
                    proposal = ut.compute_proposal_prior(
                        posterior_obs, config, prior, device
                    )

                else:
                    proposal = posterior_obs

                with timewith.TimeWith(
                    f"[TestingRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    if config["infer"]["compute_coverage"]:
                        if config["infer"]["sim_dataset"]:
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
                            test_dataset_path = str(
                                pathlib.Path().joinpath(
                                    config["test_data_loader"]["dataset_path"],
                                    f"generated_dataset/round_{i}",
                                )
                            )

                        (_, parameter, matrix,) = ut.prepare_dataset_sbi(
                            test_dataset_path, config, logger
                        )

                        # Saving the testing data to reuse it in the next rounds if the proposal is truncated with the prior.
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

                with timewith.TimeWith(
                    f"[ComputeRestrictedPriorRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    logger.info(
                        f"Sampling from the posterior for round {i + 1}..."
                    )

                    observed_samples_posterior = posterior_obs.sample(
                        (10000,), show_progress_bars=True
                    ).cpu()

                ut.corner_plot(
                    observed_samples_posterior,
                    dataset,
                    f"{save_dir_round}/corner_plot_observed_sample_{i}.pdf",
                )
                torch.save(
                    observed_samples_posterior,
                    f"{save_dir_round}/samples_posterior_{i}.pt",
                )


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

    infer(args.parse_args(), configuration)
