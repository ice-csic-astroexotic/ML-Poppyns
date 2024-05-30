"""
    Inference script for SBI.

    This script performs inference on a test dataset in a simulation-based inference framework with the sbi package.
    It loads a density estimator trained to approximate the posterior distribution for a dataset of simulated data
    and checks its performance on a test dataset.
    Simulation-based Calibration is also performed to check if the posterior is well behaved.
    See https://www.mackelab.org/sbi/ for more details.

     Running the code:

        python3 infer_sbi.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import collections
import os
import pathlib
import pickle
import sys
import time

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.analysis import check_sbc, run_sbc, sbc_rank_plot
from sbi.inference import SNPE

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loader_multichannel_array_stat as dl
from pypopsyn.learning.utils.request_device import request_device
from utilities.coverage_probability import coverage_prob


def calculate_smallest_hdr(
    posterior: callable,
    true_value: torch.tensor,
    posterior_samples: torch.tensor,
    simulation_output: torch.tensor,
    device: str,
) -> float:

    """
    Calculating the smallest highest density region of the posterior, that contains the true value.

    Args:
        posterior (Callable): Posterior distribution function.
        true_value (torch.tensor): Tensor containing the values of the parameters used to generate the simulated
        population in simulation_output.
        posterior_samples (torch.tensor): Tensor containing the samples from the inferred posterior distribution
        for simulation_output.
        simulation_output (torch.tensor): Tensor containing the maps of the simulated population.
        device (str): String specifying the type of the device used to run the script.

    Returns:
        hdr (float): Smallest highest density region of the posterior that contains the true value.
    """

    # Evaluating the PDF value of the ground truth.
    log_p_true = posterior.log_prob(
        true_value.to(device), simulation_output.to(device)
    )

    # Evaluating the PDF values of the posterior samples.
    log_p_samples = posterior.log_prob(
        posterior_samples.to(device), simulation_output.to(device)
    )

    # Determining the fraction of PDF values that are larger than that of the ground truth.
    hdr = (log_p_samples > log_p_true).sum()
    hdr = hdr / len(log_p_samples)
    return hdr


def infer(args, config):

    # Get handle for the logger --------------------------------------------
    logger = config.get_logger("Inference")
    logger.info("Logger initialized...")

    # Initialize the path where the time profiling will be saved.
    prof_log_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_log"])
    )
    prof_json_path = str(
        pathlib.Path().joinpath(config.log_dir, config["profile_json"])
    )

    # Remove the profile.json and profile.log files to prevent interrupted server connections issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)

    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

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

            # Initialize the torch seed.
            if config["set_manual_seed"] is True:
                torch.manual_seed(config["manual_seed"])
                logger.info("Seed: {}".format(config["manual_seed"]))
            else:
                torch.manual_seed(int(time.time()))
                logger.info("Seed: {}".format(int(time.time())))

            dataset_path = config["test_data_loader"]["dataset_path"]
            dataset_stat_path = config["test_data_loader"]["statistic_path"]
            filter_inputs = config["test_data_loader"]["filter_inputs"]
            filter_labels = config["test_data_loader"]["filter_labels"]
            normalize = config["test_data_loader"]["normalize"]
            standardize = config["test_data_loader"]["standardize"]
            input_shape = config["arch"]["args"]["input_shape"]
            n_components = config["density_estimator"]["args"][
                "num_components"
            ]
            n_parameters = len(filter_labels)

            # Set up GPU device if available.
            logger.info("Requesting {} GPUs...".format(config["n_gpu"]))
            device, device_ids = request_device(logger, config["n_gpu"])
            logger.info("Devices obtained: {}".format(device_ids))

        with timewith.TimeWith(
            "[TestDatasetLoader]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):

            logger.info(
                "Loading the test dataset and applying either norm or std based on the experiment..."
            )

            try:
                dataset = dl.DatasetMultichannelArray(
                    dataset_path=dataset_path,
                    statistic_path=dataset_stat_path,
                    filter_channels=filter_inputs,
                    filter_labels=filter_labels,
                    normalize=normalize,
                    standardize=standardize,
                )
            except Exception:
                logger.exception("Error: an error occurred:")
                sys.exit(1)

            parameter = np.zeros((len(dataset), n_parameters))
            matrix = np.zeros(
                (
                    len(dataset),
                    1,
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                )
            )
            for i, (x, theta) in enumerate(dataset):
                # Reshape the matrix to have the channel number at the beginning.
                x = np.moveaxis(x, -1, 0)

                if list(x.shape) != input_shape:
                    logger.error(
                        "Mismatch between the shape of the input data x {} and the input shape specified "
                        "in the configuration file {}".format(
                            x.shape, input_shape
                        )
                    )
                    sys.exit()

                matrix[i] = x
                parameter[i] = theta

            # Transform the maps and labels into torch.tensors.
            parameter = torch.from_numpy(parameter).type(torch.float32)
            matrix = torch.from_numpy(matrix).type(torch.float32)

            # Loading the test data as a data frame and extracting the ground truth labels.
            dataset_df = pd.read_csv(dataset_path)
            parameter_labels = dataset_df.columns[filter_labels]

        with timewith.TimeWith(
            "[InferenceSetup]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):

            # Set prior distribution for the parameters ------------------------------------------
            logger.info("Set prior distribution...")

            if normalize:
                # All the parameters are rescaled in the range [0, 1].
                prior = utils.BoxUniform(
                    low=torch.tensor(np.zeros(n_parameters)),
                    high=torch.tensor(np.ones(n_parameters)),
                    device=f"{device}",
                )
            elif standardize:
                # All the parameters are rescaled so that they have mean 0 and std 1.
                # We consider a range of 5 std [-5, 5].
                prior = utils.BoxUniform(
                    low=torch.tensor(-5.0 * np.ones(n_parameters)),
                    high=torch.tensor(5.0 * np.ones(n_parameters)),
                    device=f"{device}",
                )
            else:
                # Set the prior range to the range of the parameters.
                prior = utils.BoxUniform(
                    low=torch.tensor(dataset.target_min),
                    high=torch.tensor(dataset.target_max),
                    device=f"{device}",
                )

            # Set up the inference procedure -----------------------------
            logger.info("Loading the amortized trained posterior...")
            # By default the procedure uses SNPE-C
            # (https://www.mackelab.org/sbi/reference/#sbi.inference.snpe.snpe_c.SNPE_C).
            inference = SNPE()

            # Load the trained model.
            logger.info("Loading the trained model...")
            logger.info(
                "Inference is performed with the trained model: {}".format(
                    args.trained_model
                )
            )
            with open(args.trained_model, "rb") as f:
                trained_model = pickle.load(f)

            # Build the posterior.
            posterior = inference.build_posterior(
                trained_model.to(device), prior=prior
            )

        with timewith.TimeWith(
            "[Inference]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):

            # Compute the loss over the test dataset (with batch size = 1), extract the Gaussian mixture
            # coefficients and generate the corner plots.
            logger.info(
                "Computing the average loss over the test dataset, extracting Gaussian mixture coefficients,"
                "estimating the hdr for the coverage probability and generating corner plots...."
            )
            test_loss_mean = torch.tensor([0.0]).to(device)

            # Compute the coefficient for each of the Gaussian components.
            coeff_Gaussians = np.zeros((len(dataset), n_components))
            mean_Gaussians = np.zeros(
                (len(dataset), n_components, n_parameters)
            )
            precision_Gaussians = np.zeros(
                (len(dataset), n_components, n_parameters, n_parameters)
            )

            hdr_testset = np.zeros(len(dataset))

            for i in range(len(dataset)):

                test_loss_mean += posterior.log_prob(
                    parameter[i].to(device), matrix[i].to(device)
                )

                posterior_estimator = posterior.posterior_estimator

                # Extracting the latent vector, i.e., the output from the CNN, for each of the test samples.
                encoded_matrix = posterior_estimator._embedding_net(
                    matrix[i].to(device)
                )
                # Compute the parameters of each Gaussian component for each of the test samples.
                (
                    logits,
                    means,
                    precision,
                    sumlogdiag,
                    precfs,
                ) = posterior_estimator._distribution.get_mixture_components(
                    encoded_matrix
                )

                # Normalize the coefficient of each Gaussian, i.e., the sum over the coefficients is equal to 1.
                logits_norm = logits - torch.logsumexp(
                    logits, dim=-1, keepdim=True
                )
                coeff_Gaussians[i, :] = np.exp(
                    logits_norm.cpu().detach().numpy()
                )

                # Save the means and the precision matrices (inverse of the covariance matrix) of each Gaussian.
                mean_Gaussians[i, :, :] = means.cpu().detach().numpy()
                precision_Gaussians[i, :, :, :] = (
                    precision.cpu().detach().numpy()
                )

                # To calculate the coverage, we calculate the narrowest highest density region
                # containing the parameter used to generate each test sample.

                # Defining the number of samples drawn for the coverage calculation. Note that for a value of 1000,
                # the calculation takes around 1.5 seconds for each test sample.

                n_samples_coverage = 1000

                posterior_samples_coverage = posterior.set_default_x(
                    matrix[i]
                ).sample((n_samples_coverage,), show_progress_bars=False)
                smallest_hdr = calculate_smallest_hdr(
                    posterior,
                    parameter[i],
                    posterior_samples_coverage,
                    matrix[i],
                    device,
                )
                hdr_testset[i] = smallest_hdr

                # If the "corner plot" argument is set to True, we draw samples from the inferred posterior
                # distribution. Moreover, we save these samples and the corresponding corner plot.
                if args.corner_plot:

                    samples = (
                        posterior.set_default_x(matrix[i])
                        .sample((50000,), show_progress_bars=False)
                        .cpu()
                    )

                    # Save the statistics for the filtered labels.
                    par_max = torch.tensor(dataset.target_max)
                    par_min = torch.tensor(dataset.target_min)
                    par_std = torch.tensor(dataset.target_std)
                    par_mean = torch.tensor(dataset.target_mean)

                    # If the parameters were normalized or standardized rescale quantities to their physical ranges.
                    if normalize:
                        parameter[i] = (
                            parameter[i] * (par_max - par_min) + par_min
                        )
                        samples = samples * (par_max - par_min) + par_min

                    elif standardize:
                        parameter[i] = parameter[i] * par_std + par_mean
                        samples = samples * par_std + par_mean

                    # Save the samples from the inferred posterior distribution.
                    torch.save(
                        samples,
                        f"{config.log_dir}/samples_{i}.pt",
                    )

                    # Saving the best estimated parameters and the 95% CI into the log.txt file.
                    quantile = np.quantile(
                        samples, [0.025, 0.5, 0.975], axis=0
                    )

                    logger.info(
                        "Estimated parameter values (we consider the median as the best value"
                        "and the 95 % credibility interval):"
                    )

                    for s in range(n_parameters):

                        param_quantile = quantile[:, s]
                        logger.info(
                            f"{parameter_labels[s]} = {param_quantile[1]} + {param_quantile[2] - param_quantile[1]} - {param_quantile[1] - param_quantile[0]}"
                        )

                    range_param = [
                        [par_min[v], par_max[v]] for v in range(len(par_max))
                    ]

                    param_median = quantile[1, :]

                    # Corner plot of the inferred posterior distributions for each parameter.
                    figure = corner.corner(
                        samples.detach().cpu().numpy(),
                        bins=32,
                        labels=parameter_labels,
                        range=range_param,
                        quantiles=[0.025, 0.5, 0.975],
                        levels=(
                            1 - np.exp(-0.5),
                            1 - np.exp(-2),
                            1 - np.exp(-9.0 / 2.0),
                        ),  # 1, 2 and 3 sigma levels
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
                    corner.overplot_lines(
                        figure, param_median, color="tab:red"
                    )
                    corner.overplot_lines(
                        figure, parameter[i], color="tab:blue"
                    )
                    corner.overplot_points(
                        figure,
                        param_median[None],
                        marker="s",
                        color="tab:red",
                    )
                    corner.overplot_points(
                        figure,
                        parameter[None, i],
                        marker="s",
                        color="tab:blue",
                    )
                    plt.savefig(f"{config.log_dir}/corner_plot_{i}.pdf")
                    plt.close()

            logger.info("Computing the coverage probability...")

            coverage_prob(hdr_testset, 12, config.log_dir)

            # Saving the coefficients of each of the Gaussian components into a .csv file.
            df_coeff = pd.DataFrame(data=coeff_Gaussians)
            df_coeff.to_csv(f"{config.log_dir}/coeff_Gaussians.csv")

            # To save the precision and means for the whole test dataset in a numpy array uncomment the lines below.
            # np.save(f"{config.log_dir}/mean_Gaussians.npy",mean_Gaussians)
            # np.save(f"{config.log_dir}/precision_Gaussians.npy",precision_Gaussians)

            # Divide the cumulative test loss by the number of samples to obtain the average loss over the test set.
            test_loss_mean = test_loss_mean / len(dataset)
            logger.info(
                "Average loss over the test dataset: {}".format(test_loss_mean)
            )

        with timewith.TimeWith(
            "[SimulationBasedCalibration]",
            prof_log_path,
            prof_json_path,
            config["show_profiling"],
        ):

            if len(dataset) < 300:
                logger.warning(
                    "WARNING: Simulation-based Calibration cannot be performed due to the limited number of test samples."
                    "For SBC, the number of test samples should be on the order of 1000s to give reliable results. "
                    "We recommend using 10000."
                )
                sys.exit()

            else:
                logger.info("Perform Simulation-based Calibration...")

                # Run SBC: for each test sample, we draw 10000 posterior samples.
                num_posterior_samples = 10000
                ranks, dap_samples = run_sbc(
                    parameter.to(device),
                    matrix.to(device),
                    posterior,
                    num_posterior_samples=num_posterior_samples,
                )

                # Saving the ranks and the number of posterior samples to reproduce the plot.
                torch.save(ranks, f"{config.log_dir}/ranks.pt")
                logger.info(
                    f"Number of posterior samples to generate the plot of the ranks is {num_posterior_samples}"
                )

                logger.info("Check the rank statistics...")
                # Check if the rank distributions follow a uniform distribution with three different tests
                # (see [here](https://www.mackelab.org/sbi/tutorial/13_diagnostics_simulation_based_calibration/)
                # for more details on these tests).
                check_stats = check_sbc(
                    ranks,
                    parameter.to(device),
                    dap_samples.to(device),
                    num_posterior_samples=num_posterior_samples,
                    num_c2st_repetitions=5,
                )

                logger.info(
                    f"kolmogorov-smirnov p-values \n - check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"
                )

                logger.info(
                    f"c2st accuracies \n - check_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()} "
                    f"\n - check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}"
                )

                # Visually check if the ranks follow a uniform distribution.
                # The gray band represents the 99% credibility interval around the mean for a uniform distribution.
                f, ax = sbc_rank_plot(
                    ranks=ranks,
                    num_posterior_samples=num_posterior_samples,
                    plot_type="hist",
                    num_bins=30,  # When passing None the default is len(dataset_test) / 20.
                    parameter_labels=parameter_labels,
                )

                f.savefig(
                    f"{config.log_dir}/ranks_histograms.pdf",
                    bbox_inches="tight",
                )

                f, ax = sbc_rank_plot(
                    ranks=ranks,
                    num_posterior_samples=num_posterior_samples,
                    plot_type="cdf",
                    parameter_labels=parameter_labels,
                )

                f.savefig(
                    f"{config.log_dir}/ranks_cumulative.pdf",
                    bbox_inches="tight",
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="PyPopSyn Simulation-Based Inference"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="pypopsyn/learning/config_sbi.json",
        help="Configuration file path.",
    )
    args.add_argument(
        "--corner_plot",
        type=bool,
        default=False,
        help="If a posterior corner plot for each test sample is required, this argument should be set to True.",
    )

    args.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="Path to pretrained model.",
    )

    args.add_argument(
        "--infer",
        nargs="?",
        type=str,
        default=True,
        help="Flag to set up the inference saving path. If False you are in training mode.",
    )

    CustomArgs = collections.namedtuple(
        "CustomArgs", "flags type nargs target"
    )

    options = [
        CustomArgs(
            ["--dataset"],
            type=str,
            nargs="?",
            target="test_data_loader;dataset_path",
        ),
        CustomArgs(
            ["--dataset_statistics"],
            type=str,
            nargs="?",
            target="test_data_loader;statistic_path",
        ),
        CustomArgs(
            ["--filter_inputs"],
            type=int,
            nargs="*",
            target="test_data_loader;filter_inputs",
        ),
        CustomArgs(
            ["--filter_labels"],
            type=int,
            nargs="*",
            target="test_data_loader;filter_labels",
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
            ["--normalize"],
            type=bool,
            nargs="?",
            target="test_data_loader;normalize",
        ),
        CustomArgs(
            ["--standardize"],
            type=bool,
            nargs="?",
            target="test_data_loader;standardize",
        ),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        args,
        options,
    )

    infer(args.parse_args(), configuration)
