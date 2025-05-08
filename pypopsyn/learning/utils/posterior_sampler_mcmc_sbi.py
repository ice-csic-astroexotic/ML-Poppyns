"""
    Posterior Sampling Script for SNLE-Trained SBI Models

    This script samples from the posterior distribution of an SNLE-trained method and computes the log probability for
    each sample. Note that this is performed `nchain` times to create chains of posterior samples and log probability
    to latter use them with the harmonic package (https://github.com/astro-informatics/harmonic) to compute the model
    evidence at the observed data.


    Display help message to run the code:

    python posterior_sampler_mcmc_sbi.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""
import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.utils.sbi_builder as sbi_builder
import pypopsyn.learning.utils.sbi_utils as ut

# Get handle for the logger --------------------------------------------
log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def run_posterior_sampling(args: argparse.Namespace) -> None:
    """
    Samples from the posterior distribution of an SNLE-trained SBI model and computes the log-probability
    of each sample. The process is repeated across multiple MCMC chains to facilitate evidence estimation
    using the harmonic package (https://github.com/astro-informatics/harmonic).

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - exp_path (str): Path to the experiment directory containing the configuration file.
            - round (int): Round number of the SNLE training to use.
            - learning_path (str): Directory where the trained inference models are stored.
            - mcmc_sampler (str): The MCMC sampler to use for posterior sampling. Options include
              'slice_np' (stable) and 'slice_np_vectorized' (faster).
            - save_dir (str): Directory to save the sampled posterior chains and log-probabilities.
            - nsamples (int): Number of posterior samples to generate per chain.
            - nchains (int): Number of MCMC chains to generate.
            - thin_mcmc (int): Thinning parameter for the MCMC sampler. Keeps every n-th sample to reduce
                autocorrelation.
            - num_chains (int): Number of MCMC chains to run for posterior sampling.
            - device (str): Device to run the computation on ('cpu' or 'cuda').

    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    config_path = args.exp_path + "config_sbi.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    # Loading the observed sample.
    dataset, _, matrix_obs = ut.prepare_dataset_sbi(
        config["observed_sample"]["dataset_path"], config, log, True
    )
    ensemble = config["trainer"]["ensemble"]
    ensemble_size = config["trainer"]["size_ensemble"] if ensemble else 1
    learning_path = os.path.join(args.learning_path, f"round_{args.round}")

    # Load inference and initialize the prior. To extend the prior, modify prior_ranges in the configuration file.
    inference_list = sbi_builder.load_inference(
        config, args.round, args.learning_path, ensemble
    )
    prior = sbi_builder.initialize_prior(config, args.device, dataset)

    posteriors_list = []
    for index in range(ensemble_size):
        trained_model_path = (
            os.path.join(
                learning_path, f"trained_model_ensemble_{index}.pickle"
            )
            if ensemble
            else os.path.join(learning_path, "trained_model.pickle")
        )

        inference = inference_list[index if ensemble else 0]

        with open(trained_model_path, "rb") as f:
            density_estimator = pickle.load(f)

        posterior = inference.build_posterior(
            density_estimator=density_estimator.to(args.device),
            prior=prior,
            mcmc_method=args.mcmc_sampler,
            mcmc_parameters={
                "num_chains": args.num_chains,
                "thin": args.thin_mcmc,
            },
        )

        posteriors_list.append(posterior)

    if ensemble:
        ensemble_size = len(inference_list)

        # Giving each network in the ensemble an equal weight.
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list,
            weights=weights_ensemble.to(args.device),
        )
    else:
        final_posterior = posteriors_list[0]

    posterior_obs_chain = [
        final_posterior.set_default_x(matrix_obs.to(args.device)).sample(
            (args.nsamples,), show_progress_bars=True
        )
        for _ in range(args.nchains)
    ]

    log_prob_chain = [
        final_posterior.log_prob(
            posterior_obs_chain[i], matrix_obs.to(args.device)
        )
        for i in range(args.nchains)
    ]

    # Move tensors to CPU and stack.
    posterior_obs_chain = np.stack(
        [tensor.cpu().numpy() for tensor in posterior_obs_chain]
    )
    log_prob_chain = np.stack(
        [tensor.cpu().numpy() for tensor in log_prob_chain]
    )

    torch.save(posterior_obs_chain, f"{args.save_dir}/posterior_obs_chain.pt")
    torch.save(log_prob_chain, f"{args.save_dir}/log_prob_chain.pt")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Sampling posterior from SNLE model."
    )
    args.add_argument(
        "--exp_path",
        type=str,
        required=True,
        help="Path to the experiment directory.",
    )
    args.add_argument("--round", type=int, required=True, help="Round number.")
    args.add_argument(
        "--learning_path",
        type=str,
        required=True,
        help="Path where the inference and trained models are stored.",
    )
    args.add_argument(
        "--mcmc_sampler",
        type=str,
        default="slice_np",
        help="Type of MCMC sampler to use. Options: 'slice_np' (more stable) or 'slice_np_vectorized' (faster).",
    )
    args.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the generated samples and their associated log-probability values.",
    )
    args.add_argument(
        "--nsamples",
        type=int,
        default=10000,
        help="Number of posterior samples to generate.",
    )
    args.add_argument(
        "--nchains",
        type=int,
        default=4,
        help="Number of Markov Chain Monte Carlo (MCMC) chains.",
    )
    args.add_argument(
        "--thin_mcmc",
        type=int,
        default=5,
        help="Thinning parameter for the MCMC sampler. Keeps every n-th sample to reduce autocorrelation.",
    )
    args.add_argument(
        "--num_chains",
        type=int,
        default=20,
        help="Number of MCMC chains to run for posterior sampling.",
    )
    args.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the computation on ('cpu' or 'cuda').",
    )

    args = args.parse_args()

    run_posterior_sampling(args)
