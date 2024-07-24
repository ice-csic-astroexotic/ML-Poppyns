"""
    Inference script for TSNPE algorithm

    This script performs inference on a test dataset within a TSNPE framework using the sbi package. It loads the
    trained density estimator to approximate the posterior distribution for a dataset of simulated data and evaluates
    its performance on a test dataset in each round.

    Display help message to run the code:

    python infer_tsnpe.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import argparse
import collections
import os
import pathlib
import pickle
import time
from logging import Logger
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.configuration_parser as configuration_parser
import utilities.benchmark.timewith as timewith
from pypopsyn.learning.train_tsnpe import (
    compute_proposal_prior,
    compute_rank_coverage,
    corner_plot,
    initialize_inference,
    prepare_dataset_sbi,
)
from pypopsyn.learning.utils.request_device import request_device


def load_posterior(
    config: configuration_parser.ConfigurationParser,
    save_dir_round: pathlib.Path,
    logger: Logger,
    inference_list: Union[SNPE_C, List[SNPE_C]],
    device: torch.device,
    round_current: int,
) -> Union[DirectPosterior, NeuralPosteriorEnsemble]:
    """
    Load the trained density estimator for a given round.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying training parameters.
        save_dir_round (pathlib.Path): Directory where the trained model will be saved or is saved already.
        logger (Logger): Logger object.
        inference_list (Union[SNPE_C, List[SNPE_C]]): sbi inference object or list of inference objects for ensemble.
        device (torch.device): Device used for training.
        round_current (int): Current round number.

    Returns:
        (Union[DirectPosterior, NeuralPosteriorEnsemble]): Trained density estimator or ensemble of estimators.
    """
    ensemble = config["trainer"]["ensemble"]
    ensemble_size = config["trainer"]["size_ensemble"] if ensemble else 1

    posteriors_list = []

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

        posterior = inference.build_posterior(density_estimator.to(device))
        posteriors_list.append(posterior)

    if ensemble:
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list, weights=weights_ensemble.to(device)
        )
    else:
        final_posterior = posteriors_list[0]

    return final_posterior


def infer(
    args: argparse.Namespace, config: configuration_parser.ConfigurationParser
) -> None:
    """
    Infer the posterior distribution for the observed population using the truncated sequential neural posterior
    estimator approach described in Deistler et al. (2022), assuming that the train_tsnpe.py script has already been run
    and a trained_model.pkl was generated.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
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
            # Load the training dataset to access the statistics. Note that when performing inference, we do not need
            # to use the training dataset.
            dataset, _, _ = prepare_dataset_sbi(
                train_dataset_path, config, logger
            )
            n_parameters = len(torch.tensor(config["prior_ranges"]["low"]))
            num_rounds = config["trainer"]["num_rounds"]

            # Loading the header of the train dataset to extract the ground truth labels.
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

            # Create the matrix for the observed sample of neutron stars.
            _, _, x_o = prepare_dataset_sbi(
                config["observed_sample"]["dataset_path"],
                config,
                logger,
                atnf=True,
            )

            # During inference, we load the trained model. The inference object is used to identify which neural
            # posterior estimation algorithm is employed. In this case we use SNPE. Therefore, we only need to
            # initialize the network at the beginning.
            inference_list = initialize_inference(
                config, device, prior, ensemble
            )
        for i in range(num_rounds):

            save_dir_round = config.log_dir / f"round_{i}"
            save_dir_round.mkdir(parents=True, exist_ok=True)
            # Load_dir folder is where the trained_model.pkl is saved.
            load_dir = config["infer"]["load_dir"]
            load_dir_round = load_dir + f"/round_{i}"

            with timewith.TimeWith(
                f"[TotalRound{i}]",
                prof_log_path,
                prof_json_path,
                config["show_profiling"],
            ):

                posterior = load_posterior(
                    config, load_dir_round, logger, inference_list, device, i
                )

                with timewith.TimeWith(
                    f"[TestingRound{i}]",
                    prof_log_path,
                    prof_json_path,
                    config["show_profiling"],
                ):
                    if config["infer"]["compute_coverage"]:

                        test_dataset_path = str(
                            pathlib.Path().joinpath(
                                config["test_data_loader"]["dataset_path"],
                                f"generated_dataset/round_{i}",
                            )
                        )

                        _, parameter_test, matrix_test = prepare_dataset_sbi(
                            test_dataset_path, config, logger
                        )
                        logger.info(
                            f"Computing the ranks and the coverage probability for round_{i}"
                        )

                        compute_rank_coverage(
                            save_dir=save_dir_round,
                            parameter=parameter_test,
                            matrix=matrix_test,
                            posterior=posterior,
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
                        f"Computing the proposal prior for round {i + 1}..."
                    )

                    posterior_obs = posterior.set_default_x(x_o)
                    proposal = compute_proposal_prior(
                        posterior_obs, config, prior, device
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
                            f"{save_dir_round}/corner_plot_prior_round_{i + 1}.pdf",
                        )
                        # Save the samples from the inferred posterior distribution.
                        torch.save(
                            observed_samples_proposal,
                            f"{save_dir_round}/samples_prior_{i + 1}.pt",
                        )

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

    infer(args.parse_args(), configuration)
