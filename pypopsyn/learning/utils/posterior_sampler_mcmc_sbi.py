"""
    Posterior Sampling Script for SNLE-Trained SBI Models

    This script samples from the posterior distribution of an SNLE-trained method and computes the log probability for each sample. Note that this is perform nchain times to create a chain of posterior samples and log probability to latter use them with the harmonic package to compute the model evidence at the observed data.

    Display help message to run the code:

    python sampling_posterior_snle.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""
import argparse
import json
import logging
import sys

import numpy as np
import torch
from sbi import utils
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

import pypopsyn.learning.utils.sbi_builder as sbi_builder
import pypopsyn.learning.utils.sbi_utils as ut

# Get handle for the logger --------------------------------------------
log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def run_posterior_sampling(args: argparse.Namespace) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    config_path = args.exp_path + "config_snle.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    # Loading the observed sample.
    dataset, _, matrix_obs = ut.prepare_dataset_sbi(
        config["observed_sample"]["dataset_path"], config, log, True
    )
    ensemble = config["trainer"]["ensemble"]
    inference_list = sbi_builder.load_inference(
        config, args.round, args.learning_path, ensemble
    )
    device = "cuda"

    low = (
        torch.tensor(config["prior_ranges"]["low"]) - dataset.target_mean
    ) / dataset.target_std
    high = (
        torch.tensor(config["prior_ranges"]["high"]) - dataset.target_mean
    ) / dataset.target_std
    prior = utils.BoxUniform(
        low=low,
        high=high,
        device=f"{device}",
    )

    posteriors_list = []
    for inference in inference_list:
        posterior = inference.build_posterior(
            prior=prior,
            mcmc_method=args.mcmc_sampler,
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        posteriors_list.append(posterior)
    if ensemble:
        ensemble_size = len(inference_list)

        # Giving each network in the ensemble an equal weight.
        weights_ensemble = torch.ones(ensemble_size) / ensemble_size
        final_posterior = NeuralPosteriorEnsemble(
            posteriors_list,
            weights=weights_ensemble.to(device),
        )
    else:
        final_posterior = posteriors_list[0]

    posterior_obs_chain = [
        final_posterior.set_default_x(matrix_obs.to("cuda")).sample(
            (args.nsamples,), show_progress_bars=True
        )
        for _ in range(args.nchains)
    ]

    log_prob_chain = [
        final_posterior.log_prob(posterior_obs_chain[i], matrix_obs.to("cuda"))
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
    args = args.parse_args()

    run_posterior_sampling(args)
