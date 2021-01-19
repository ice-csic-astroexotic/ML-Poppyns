#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Plot two parameters inference results.

    This script reads the file .csv containing the inference results of a network trained
    on two parameters and plots the results in the form of residual plots.

    Running the code:

        python3 scripts/plot_inference_result_2p.py --inference_file filename.csv --save_dir inference_result

        or

        python3 scripts/plot_inference_result_2p.py --h

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

import utilities.plot_settings
import utilities.statistics as stat


def plot_inference_results(args) -> None:
    """
    This method reads the file .csv containing the inference results of a network trained
    on two parameters and plots the results in the form of residual plots.
    The file have to contain the prediction results for the sigma_k and h_c parameters respectively.

    Args:
        inference_file (str): Path to where the file containing the inference results is stored.

        save_dir (str): Path to where the generated plots will be saved.
    """

    # Read the file.
    data = pd.read_csv(f"{args.inference_file}")
    target_sigmak = data["target:sigma_k"].to_numpy()
    prediction_sigmak = data["predicted:sigma_k"].to_numpy()
    target_hc = data["target:h_c"].to_numpy()
    prediction_hc = data["predicted:h_c"].to_numpy()

    # Compute the RMSE and MRE values for the total inferred datasets.
    RMSE_sigmak = np.sqrt(np.mean((prediction_sigmak - target_sigmak) ** 2))
    RMSE_hc = np.sqrt(np.mean((prediction_hc - target_hc) ** 2))

    MRE_sigmak = np.mean(
        abs((prediction_sigmak - target_sigmak) / target_sigmak)
    )
    MRE_hc = np.mean(abs((prediction_hc - target_hc) / target_hc))

    # Compute the RMSE values as a function of the target values.
    n_bins = 50
    (
        bin_centers_sigmak,
        running_rmse_sigmak,
        running_average_sigmak,
        running_mre_sigmak,
    ) = stat.inference_running_stat(prediction_sigmak, target_sigmak, n_bins)

    (
        bin_centers_hc,
        running_rmse_hc,
        running_average_hc,
        running_mre_hc,
    ) = stat.inference_running_stat(prediction_hc, target_hc, n_bins)

    # Plot predicted vs target values.
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

    plt.savefig(f"{args.save_dir}/2par_pred_vs_target_sigmak.pdf")

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

    plt.savefig(f"{args.save_dir}/2par_pred_vs_target_hc.pdf")

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

    plt.savefig(f"{args.save_dir}/2par_rmse_sigmak.pdf")

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

    plt.savefig(f"{args.save_dir}/2par_rmse_hc.pdf")

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

    plt.savefig(f"{args.save_dir}/2par_mre_sigmak.pdf")

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

    plt.savefig(f"{args.save_dir}/2par_mre_hc.pdf")

    # Plot correlation between the two parameters.
    fig, ax = plt.subplots()

    ax.set_xlabel(r"Residuals $\sigma_{\rm k}$ [km s$^{-1}$]")
    ax.set_ylabel(r"Residuals $h_{\rm c}$ [kpc]")

    ax.scatter(
        prediction_sigmak - target_sigmak,
        prediction_hc - target_hc,
        linestyle="None",
        marker="o",
        facecolors="black",
        edgecolors="None",
        s=10,
        alpha=0.3,
        rasterized=True,
    )

    ax.axhline(y=0.0, c="tab:red", linewidth=3, zorder=1)
    ax.axvline(x=0.0, c="tab:red", linewidth=3, zorder=1)

    plt.savefig(f"{args.save_dir}/residuals_correlation.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--inference_file",
        nargs="?",
        type=str,
        required=True,
        help="Path to where the file containing the inference results is stored.",
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
