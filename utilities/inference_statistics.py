"""
    Module containing statistic tools to use when analyzing the outcome of the simulations or machine learning results.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import numpy as np


def inference_running_stat(
    x: np.ndarray, targets: np.array, n_bins: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the variation of the root-mean-square error (RMSE) the mean relative error (MRE) and of the average
    residual with sign of the predicted values x by a trained neural network model over the range of the targets'
    data.

    Args:
        x (np.ndarray): predicted values.
        targets (np.ndarray): target values.
        n_bins (int): number of bins.

    Returns:
        (np.array, np.array, np.array, np.array): array of central values of each bin, running value of the RMSE
        corresponding to each bin, running value of the average residuals corresponding to each bin and running
        MRE corresponding to each bin.
    """
    inf_lim = np.min(targets)
    sup_lim = np.max(targets)
    bin_edges = np.linspace(inf_lim, sup_lim, n_bins + 1)

    # Compute the bin center values.
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Compute statistics.
    running_rmse = np.zeros(len(bin_centers))
    running_average = np.zeros(len(bin_centers))
    running_mre = np.zeros(len(bin_centers))

    for i in range(1, len(bin_edges)):
        cond = (targets > bin_edges[i - 1]) & (targets <= bin_edges[i])
        running_rmse[i - 1] = np.sqrt(np.mean((x[cond] - targets[cond]) ** 2))
        running_average[i - 1] = np.mean((x[cond] - targets[cond]))
        running_mre[i - 1] = np.mean(
            abs(x[cond] - targets[cond]) / targets[cond]
        )

    return bin_centers, running_rmse, running_average, running_mre
