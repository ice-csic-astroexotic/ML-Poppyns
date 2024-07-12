""""
    Computing the coverage probability.

    This module computes the coverage probability given the minimum highest density region for each of the test samples.
    It saves both the numpy array and the plot of the coverage probability in the `save_dir` folder.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def coverage_prob(
    hdr_testset: np.ndarray, n_betas: int, save_dir: pathlib.Path
) -> None:
    """
    Computing the coverage probability for the test dataset.
    The coverage probability array and corresponding plot will be saved in the directory specified by save_dir.

    Args:
        hdr_testset (np.ndarray): array containing the highest density region for each of the test samples.
        n_betas (int): number of betas that we want to use to compute the coverage probability.
        save_dir (pathlib.Path): path to the directory where the coverage probability array and plot will be saved.
    """

    # Calculate the coverage from the smallest hdr.
    betas = np.linspace(0, 1, n_betas)
    coverage_probability = []
    hdr_testset_sorted = np.sort(np.asarray(hdr_testset))

    # For each value of beta, we calculate the percentage of test samples for which the
    # highest density region (HDR), in hdr_testset, is lower than beta. Then, among these
    # test samples, each of the true values will fall inside this beta's HDR.

    for beta in betas:
        coverage_probability.append((hdr_testset_sorted < beta).mean())

    # Saving the coverage probability array to produce the coverage plot.
    np.save(f"{save_dir}/coverage_probability.npy", coverage_probability)

    # Plot the coverage.
    plt.plot(
        betas,
        coverage_probability,
        color="steelblue",
        label="upper right",
    )
    plt.plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel(r"Credibility level $1-\alpha$", fontsize=10)
    plt.ylabel(r"Coverage probability", fontsize=10)
    plt.savefig(f"{save_dir}/coverage_plot.pdf")
    plt.close()
