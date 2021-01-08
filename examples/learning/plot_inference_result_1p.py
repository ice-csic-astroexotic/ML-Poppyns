#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Plot single parameter inference results.

    This script reads the files .csv containing the inference results of a network trained
    on a single parameter and plots the results in the form of residual plots.

    Running the code:

        python3 scripts/plot_inference_result_1p.py --sigmak_inference_file filename.csv
        --hc_inference_file filename.csv --save_dir inference_result

        or

        python3 scripts/plot_inference_result_1p.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2020
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "Liberation serif"
rcParams["font.size"] = "30"
# rcParams['font.weight']='bold'
rcParams["figure.figsize"] = "8.0, 7.0"
rcParams["figure.autolayout"] = True

rcParams["axes.linewidth"] = "1.7"
rcParams["axes.labelpad"] = "15.0"
rcParams["axes.titlepad"] = "15.0"

rcParams["xtick.direction"] = "in"
rcParams["xtick.top"] = True
rcParams["xtick.major.pad"] = "10.0"
rcParams["xtick.minor.pad"] = "10.0"
rcParams["xtick.major.size"] = "10.0"
rcParams["xtick.major.width"] = "1.7"
rcParams["xtick.minor.size"] = "5.0"
rcParams["xtick.minor.width"] = "1.7"
rcParams["xtick.labelsize"] = "30"

rcParams["ytick.direction"] = "in"
rcParams["ytick.right"] = True
rcParams["ytick.major.pad"] = "10.0"
rcParams["ytick.minor.pad"] = "10.0"
rcParams["ytick.major.size"] = "10.0"
rcParams["ytick.major.width"] = "1.7"
rcParams["ytick.minor.size"] = "5.0"
rcParams["ytick.minor.width"] = "1.7"
rcParams["ytick.labelsize"] = "30"


def running_stat(
    x: np.ndarray, targets: np.array, n_bins: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Calculate the variation of the root mean square error (RMSE) the mean relative error (MRE) and of the mean
        residual with sign of the predicted values x over the range of the targets data.

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
        cond = (targets > bin_edges[i - 1]) & (targets < bin_edges[i])
        running_rmse[i - 1] = np.sqrt(np.mean((x[cond] - targets[cond]) ** 2))
        running_average[i - 1] = np.mean((x[cond] - targets[cond]))
        running_mre[i - 1] = np.mean(
            abs(x[cond] - targets[cond]) / targets[cond]
        )

    return bin_centers, running_rmse, running_average, running_mre


def plot_inference_results(args) -> None:
    """
    This method reads the two files .csv containing the inference results of a network trained
    on a single parameter and plots the results in the form of residual plots.
    The two files have to contain the prediction results for the sigma_k and h_c parameters respectively.

    Args:
        sigmak_inference_file (str): Path to where the file containing the inference results on the sigma_k parameter is stored.

        hc_inference_file (str): Path to where the file containing the inference results on the h_c parameter is stored.

        save_dir (str): Path to where the generated plots will be saved.
    """

    if args.sigmak_inference_file is not None:

        # Read the file.
        data = pd.read_csv(f"{args.sigmak_inference_file}")
        target_sigmak = data["target:sigma_k"].to_numpy()
        prediction_sigmak = data["predicted:sigma_k"].to_numpy()

        # Compute the RMSE and MRE values for the total inferred datasets.
        RMSE_sigmak = np.sqrt(
            np.mean((prediction_sigmak - target_sigmak) ** 2)
        )
        MRE_sigmak = np.mean(
            abs((prediction_sigmak - target_sigmak) / target_sigmak)
        )

        # Compute the statistics as a function of the target values.
        n_bins = 50
        (
            bin_centers_sigmak,
            running_rmse_sigmak,
            running_average_sigmak,
            running_mre_sigmak,
        ) = running_stat(prediction_sigmak, target_sigmak, n_bins)

        # Plot prediction vs target values.
        fig, ax = plt.subplots()

        ax.set_xlabel(r"Target $\sigma_{\rm k}$ [km s$^{-1}$]")
        ax.set_ylabel(r"Predicted $\sigma_{\rm k}$ [km s$^{-1}$]")

        ax.scatter(
            target_sigmak,
            prediction_sigmak,
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            target_sigmak,
            target_sigmak,
            linestyle="-",
            linewidth=3,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )

        plt.savefig(f"{args.save_dir}/1par_pred_vs_target_sigmak.pdf")

        # Plot residuals.
        fig, ax = plt.subplots()

        ax.set_xlabel(r"$\sigma_{\rm k, T}$ [km s$^{-1}$]")
        ax.set_ylabel(r"$\sigma_{\rm k, P} - \sigma_{\rm k, T}$ [km s$^{-1}$]")

        ax.scatter(
            target_sigmak,
            prediction_sigmak - target_sigmak,
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            np.linspace(1.0, 700.0, n_bins),
            np.zeros(n_bins),
            linestyle="-",
            linewidth=3,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            np.linspace(1.0, 700.0, n_bins),
            np.zeros(n_bins) - RMSE_sigmak,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            np.linspace(1.0, 700.0, n_bins),
            np.zeros(n_bins) + RMSE_sigmak,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            bin_centers_sigmak,
            running_average_sigmak,
            linestyle="-",
            linewidth=5,
            color="tab:blue",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.fill_between(
            bin_centers_sigmak,
            np.zeros(len(bin_centers_sigmak)) - running_rmse_sigmak,
            np.zeros(len(bin_centers_sigmak)) + running_rmse_sigmak,
            facecolors="tab:orange",
            edgecolors="None",
            alpha=0.5,
            rasterized=True,
            zorder=0,
        )

        plt.savefig(f"{args.save_dir}/1par_rmse_sigmak.pdf")

        # Plot absolute relative residuals.
        fig, ax = plt.subplots()

        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_{\rm k, T}$ [km s$^{-1}$]")
        ax.set_ylabel(
            r"$\left| (\sigma_{\rm k, P} - \sigma_{\rm k, T}) / \sigma_{\rm k, T} \right|$"
        )

        ax.scatter(
            target_sigmak,
            abs((prediction_sigmak - target_sigmak) / target_sigmak),
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            np.linspace(1.0, 700.0, 50),
            np.zeros(50) + MRE_sigmak,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            bin_centers_sigmak,
            np.zeros(len(bin_centers_sigmak)) + running_mre_sigmak,
            linestyle="-",
            linewidth=5,
            color="tab:blue",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )

        plt.savefig(f"{args.save_dir}/1par_mre_sigmak.pdf")

    if args.hc_inference_file is not None:

        # Read the file.
        data = pd.read_csv(f"{args.hc_inference_file}")
        target_hc = data["target:h_c"].to_numpy()
        prediction_hc = data["predicted:h_c"].to_numpy()

        # Compute the RMSE and MRE values for the total inferred datasets.
        RMSE_hc = np.sqrt(np.mean((prediction_hc - target_hc) ** 2))
        MRE_hc = np.mean(abs((prediction_hc - target_hc) / target_hc))

        # Compute the statistics as a function of the target values.
        n_bins = 50
        (
            bin_centers_hc,
            running_rmse_hc,
            running_average_hc,
            running_mre_hc,
        ) = running_stat(prediction_hc, target_hc, n_bins)

        # Plot predicted vs target values.
        fig, ax = plt.subplots()

        ax.set_xlabel(r"Target $h_{\rm c}$ [kpc]")
        ax.set_ylabel(r"Predicted $h_{\rm c}$ [kpc]")

        ax.scatter(
            target_hc,
            prediction_hc,
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            target_hc,
            target_hc,
            linestyle="-",
            linewidth=3,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )

        plt.savefig(f"{args.save_dir}/1par_pred_vs_target_hc.pdf")

        # Plot residuals.
        fig, ax = plt.subplots()

        ax.set_xlabel(r"$h_{\rm c, T}$ [kpc]")
        ax.set_ylabel(r"$h_{\rm c, P} - h_{\rm c, T}$ [kpc]")

        ax.scatter(
            target_hc,
            prediction_hc - target_hc,
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            np.linspace(0.02, 2.0, n_bins),
            np.zeros(n_bins),
            linestyle="-",
            linewidth=3,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            np.linspace(0.02, 2.0, n_bins),
            np.zeros(n_bins) - RMSE_hc,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            np.linspace(0.02, 2.0, n_bins),
            np.zeros(n_bins) + RMSE_hc,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            bin_centers_hc,
            running_average_hc,
            linestyle="-",
            linewidth=5,
            color="tab:blue",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.fill_between(
            bin_centers_hc,
            np.zeros(len(bin_centers_hc)) - running_rmse_hc,
            np.zeros(len(bin_centers_hc)) + running_rmse_hc,
            facecolors="tab:orange",
            edgecolors="None",
            alpha=0.5,
            rasterized=True,
            zorder=0,
        )

        plt.savefig(f"{args.save_dir}/1par_rmse_hc.pdf")

        # Plot absolute relative residuals.
        fig, ax = plt.subplots()

        ax.set_yscale("log")
        ax.set_xlabel(r"$h_{\rm c, T}$ [kpc]")
        ax.set_ylabel(
            r"$\left| ( h_{\rm c, P} - h_{\rm c, T} ) / h_{\rm c, T} \right|$"
        )

        ax.scatter(
            target_hc,
            abs((prediction_hc - target_hc) / target_hc),
            linestyle="None",
            marker="o",
            facecolors="black",
            edgecolors="None",
            s=10,
            alpha=0.3,
            rasterized=True,
        )
        ax.plot(
            np.linspace(0.02, 2.0, 50),
            np.zeros(50) + MRE_hc,
            linestyle="--",
            linewidth=5,
            color="tab:red",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            bin_centers_hc,
            np.zeros(len(bin_centers_hc)) + running_mre_hc,
            linestyle="-",
            linewidth=5,
            color="tab:blue",
            alpha=1.0,
            rasterized=True,
            zorder=1,
        )

        plt.savefig(f"{args.save_dir}/1par_mre_hc.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--sigmak_inference_file",
        nargs="?",
        type=str,
        default=None,
        help="Path to where the file containing the inference results on the sigma_k parameter is stored.",
    )
    parser.add_argument(
        "--hc_inference_file",
        nargs="?",
        type=str,
        default=None,
        help="Path to where the file containing the inference results on the h_k parameter is stored.",
    )
    parser.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="examples/inference_plots",
        help="Path to the folder where the plots will be saved.",
    )

    args = parser.parse_args()

    plot_inference_results(args)
