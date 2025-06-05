"""
    Builder of sbi.

    Display help message to run the code:

    python sbi_builder.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""


import os
import pathlib
import pickle
import sys
from logging import Logger
from typing import List, Union

import numpy as np
import torch
from sbi import utils
from sbi.inference import SNLE, SNPE, SNRE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snle.snle_a import SNLE_A
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.inference.snre.snre_b import SNRE_B
from sbi.utils import BoxUniform
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loader_multichannel_array as dl
import pypopsyn.learning.models.models as learning_models
import pypopsyn.learning.utils.sbi_utils as ut
import utilities.benchmark.timewith as timewith


def load_inference(
    config: configuration_parser.ConfigurationParser,
    round_number: int,
    save_dir: pathlib.Path,
    ensemble: bool = False,
) -> Union[List[SNPE], List[SNLE]]:
    """
    Load inference objects from pickle files. Note that this is used when resume mode is enabled
    or when doing inference with SNLE.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        round_number (int): The round number to load the inference from.
        save_dir (pathlib.Path): The directory to load the inference from.
        ensemble (bool): Flag indicating if ensemble mode is enabled. Defaults to False.

    Returns:
        (Union[List[SNPE], List[SNLE]]):  A list of inference objects.
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
                "To use the resume mode or inference with SNLE, you need to specify the correct path."
            )

        with open(inference_path, "rb") as inference_file:
            inference = pickle.load(inference_file)

        inference_list.append(inference)

    return inference_list


def compute_proposal_prior(
    posterior_obs: DirectPosterior,
    config: configuration_parser.ConfigurationParser,
    prior: utils.BoxUniform,
    device: torch.device,
) -> utils.RestrictedPrior:
    """
    Compute the proposal prior by restricting the prior to the regions where the posterior of the
    observation has non-negligible probability mass.

    Args:
        posterior_obs (DirectPosterior): Posterior distribution at the observation.
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        prior (utils.BoxUniform): Prior distribution.
        device (torch.device): Device used to run the script.

    Returns:
        (utils.RestrictedPrior): The restricted prior based on the posterior distribution of the observation.
    """
    # Computing the region of the posterior distribution used to constrain the prior. The quantile value is the default
    # in sbi. Instead of using the default 100,000 for the sample number, we sample 10,000 times which is possible as
    # our posteriors are Gaussian-like. This reduction also makes things faster.
    accept_reject_fn = utils.get_density_thresholder(
        posterior_obs,
        quantile=1e-4,
        num_samples_to_estimate_support=10000,
    )
    # Computing the new proposal by restricting the prior to the posterior of the observation.
    # If config["sir"] is set to true, the restricted prior sampling uses sampling importance
    # resampling (Rubin et al., 1988); otherwise, it employs rejection sampling. Note that the latter
    # method may take longer for a narrower posterior distribution where the rejection rate is high.
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


def build_inference_network(
    model_type: str,
    logger: Logger,
    config: configuration_parser.ConfigurationParser,
    device: torch.device,
    prior: utils.BoxUniform,
) -> Union[SNPE_C, SNLE_A, SNRE_B]:

    """
    Builds an inference object (SNPE, SNLE, or SNRE) based on the selected model_type,
    using the provided configuration, device, and prior.

    Args:
        model_type (str): The inference model_type to use. Must be one of:
                      "snpe", "snle", or "snre".
        logger (Logger): Logger object.
        config (ConfigurationParser): Configuration object that defines the architecture
                                      and training parameters for the model.
        device (torch.device): The device (CPU or GPU) on which to build and run the model.
        prior (utils.BoxUniform): The prior distribution over the parameters.

    Returns:
        Union[SNPE_C, SNLE_A, SNRE_A]: An instance of the corresponding sbi inference class,
                                       depending on the model_type specified.
    """

    if model_type.lower() == "snpe":

        # If config["compression_input"]["use_compression"] is False, input data compression will be performed directly
        # within the density network. In this case, the first component of the network is a CNN that compresses the
        # input data, and it is trained jointly with the density estimator. Note that this option is only compatible
        # with the NPE.
        posterior_nn_args = {
            "model": config["density_estimator"]["type"],
            "hidden_features": config["density_estimator"]["args"][
                "hidden_features"
            ],
            "num_components": config["density_estimator"]["args"][
                "num_components"
            ],
            "device": device,
        }

        if not config["compression_input"]["use_compression"]:
            # Build the embedding network
            embedding_net = config.init_object("arch", learning_models)

            # Initialize weights
            weight_initializer = config.init_object(
                "weights_initializer", learning_initializers
            )
            embedding_net.apply(weight_initializer)

            posterior_nn_args["embedding_net"] = embedding_net

        # The default density estimator has 3 hidden layers with a number of neurons = hidden_features.
        # The weights are initialized with the default initialization provided by pytorch.
        neural_posterior = utils.posterior_nn(**posterior_nn_args)

        # Setting up the inference procedure.
        inference = SNPE(
            density_estimator=neural_posterior,
            device=f"{device}",
            prior=prior,
        )

        return inference

    if model_type.lower() == "snle":
        inference = SNLE(
            density_estimator=config["density_estimator"]["type"],
            device=f"{device}",
            prior=prior,
        )

        return inference

    if model_type.lower() == "snre":
        inference = SNRE(
            classifier=config["density_estimator"]["classifier_nre"],
            device=f"{device}",
            prior=prior,
        )

        return inference
    else:
        logger.exception(
            "The model type '{}' is not supported. ".format(model_type)
        )
        sys.exit(1)


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
        config (configuration_parser.ConfigurationParser): Configuration object specifying the settings.
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
        inference = build_inference_network(
            model_type, logger, config, device, prior
        )
        inference_list.append(inference)

    return inference_list


def train_posterior(
    config: configuration_parser.ConfigurationParser,
    save_dir_round: pathlib.Path,
    logger: Logger,
    inference_list: Union[List[SNPE], List[SNLE]],
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

    If resume is set to True in the configuration file, this mode allows training to continue from the last completed
    round if interrupted. It uses the previously saved state to resume training without starting over.
    If ensemble is set to True in the configuration file, multiple models (an ensemble) are trained and their
    predictions are combined to ensure conservative coverages. Each of the neural networks will be trained on the same
    training dataset.

    Note that the inference object should be different for each component of the ensemble to ensure independent weights
    for each component. Moreover, if config['trainer']['model_type'] == 'snle', then an MCMC sampler is needed to sample
    from the posterior distribution.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        save_dir_round (pathlib.Path): Directory where the trained model will be saved or is saved already.
        logger (Logger): Logger object.
        inference_list (Union[List[SNPE], List[SNLE]]): List of inference sbi objects.
        parameter_round (torch.Tensor): Tensor containing the parameters for the current round.
        matrix_round (torch.Tensor): Tensor containing the matrices for the current round.
        device (torch.device): Device used to run the script.
        round_current (int): Current round number. This parameter starts at zero.
        prof_json_path (str): The profile.json path.
        prof_log_path (str): The profile.log path.
        retrain_from_scratch (bool): Whether to retrain the conditional density estimator for the posterior from
            scratch each round. Default value is False.
        proposal (DirectPosterior): The proposal prior used in the current round.

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

                # When using SNPE with a truncated prior, we set force_first_round_loss = True in the following to
                # disable the loss-function correction. Otherwise, the loss would be corrected using the proposal prior
                # during training. In the case where we do not truncate the prior and account for the correction, we
                # then pass the proposal prior to the append_simulations function.

                if (
                    config["trainer"]["truncated_prior"]
                    and model_type == "snpe"
                ):
                    train_args["force_first_round_loss"] = True

                kwargs = {}
                if (
                    model_type == "snpe"
                    and not config["trainer"]["truncated_prior"]
                ):
                    kwargs["proposal"] = proposal

                density_estimator = inference.append_simulations(
                    parameter_round.to(device),
                    matrix_round.to(device),
                    **kwargs,
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
        elif model_type == "snle" or "snre":
            posterior = inference.build_posterior(
                density_estimator=density_estimator.to(device),
                mcmc_method=config["mcmc_sampler"]["type"],
                mcmc_parameters={
                    "num_chains": config["mcmc_sampler"]["num_chains"],
                    "thin": config["mcmc_sampler"]["thin"],
                },
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

        # Saving the training statistics. If resuming in the first round, no training is performed, i.e., nothing is
        # saved.
        if not resume or round_current != 0:
            ut.save_training_statistics(
                config, inference, index, effective_round
            )

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
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        device (torch.device): Device used to run the script.
        dataset (DatasetMultichannelArray): Dataset where the statistics are saved.

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
    inference_list: Union[List[SNPE], List[SNLE]],
    device: torch.device,
    round_current: int,
) -> Union[DirectPosterior, NeuralPosteriorEnsemble]:
    """
    Load the trained density estimator for a given round.

    Args:
        config (configuration_parser.ConfigurationParser): Configuration object specifying the model settings.
        logger (Logger): Logger object.
        inference_list (Union[List[SNPE], List[SNLE]]): List of sbi inference object.
        device (torch.device): Device used to run the script.
        round_current (int): Current round number. This parameter starts at zero.

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
                density_estimator=density_estimator.to(device),
                mcmc_method=config["mcmc_sampler"]["type"],
                mcmc_parameters={
                    "num_chains": config["mcmc_sampler"]["num_chains"],
                    "thin": config["mcmc_sampler"]["thin"],
                },
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
