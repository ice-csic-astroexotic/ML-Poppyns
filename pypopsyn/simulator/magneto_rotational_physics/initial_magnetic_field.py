"""
    Initial probability distribution of pulsar magnetic fields.

    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np

import utilities.samplers.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg


def pdf_log10_magnetic_field_2normal(log10B: np.ndarray) -> np.ndarray:
    """
    Mixture of two Gaussian distributions for the logarithm log10 of neutron stars' initial magnetic field.
    The mean and standard deviation are defined in the configuration file.

    Args:
        log10B (np.ndarray): log10 of the magnetic field Strength in [G].

    Returns:
        (np.ndarray): value of the pdf for each log10B.
    """

    # Define the dispersions of the two Gaussian components.
    mean_1 = cfg["B_initial_log10_mean1"]
    mean_2 = cfg["B_initial_log10_mean2"]
    sigma_1 = cfg["B_initial_log10_sigma1"]
    sigma_2 = cfg["B_initial_log10_sigma2"]
    # Define the fractional contribution of the first Gaussian.
    w = cfg["B_initial_weight"]

    pdf_gaussian_1 = (
        1.0
        / (np.sqrt(2 * np.pi) * sigma_1)
        * np.exp(-((log10B - mean_1) ** 2) / (2.0 * sigma_1**2))
    )

    pdf_gaussian_2 = (
        1.0
        / (np.sqrt(2 * np.pi) * sigma_2)
        * np.exp(-((log10B - mean_2) ** 2) / (2.0 * sigma_2**2))
    )

    pdf = w * pdf_gaussian_1 + (1.0 - w) * pdf_gaussian_2

    return pdf


def pdf_gaussian_custom_norm(
    x: np.ndarray, mean: float, sigma: float, norm: float
) -> np.ndarray:
    """
    A Gaussian function with custom normalization.

    Args:
        x (np.ndarray): array of values where to compute the function.
        mean (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        norm (float): Normalization of the Gaussian distribution.

    Returns:
        (np.ndarray): value of the Gaussian distribution for each x.
    """
    gauss_func = norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    return gauss_func


def pdf_log10_magnetic_field_smooth_tophat(
    log10B: np.ndarray,
) -> np.ndarray:
    """
    A smooth top-hat function with a sloped central region and gaussian rise and decay.
    The parameters of this distribution are defined in the configuration file.

    Args:
        log10B (np.ndarray): log10 of the magnetic field Strength in [G].

    Returns:
        (np.ndarray): value of the pdf for each log10B.
    """
    # Define the parameters of the distribution.
    rise_mean = cfg["B_initial_log10_rise_mean"]
    rise_sigma = cfg["B_initial_log10_rise_sigma"]
    decay_mean = cfg["B_initial_log10_decay_mean"]
    decay_sigma = cfg["B_initial_log10_decay_sigma"]
    slope = cfg["B_initial_log10_slope"]

    # Calculate the x coordinate of the center of the sloped part.
    center = (rise_mean + decay_mean) / 2

    # Evaluate the heights of the two Gaussians assuming that the slope pass through the point with coordinates (center, 1).
    norm_rise_gaussian = slope * (rise_mean - center) + 1
    norm_decay_gaussian = slope * (decay_mean - center) + 1

    # Gaussian rise on the left side.
    rise_gaussian = pdf_gaussian_custom_norm(
        log10B, rise_mean, rise_sigma, norm_rise_gaussian
    )

    # Gaussian decay on the right side.
    decay_gaussian = pdf_gaussian_custom_norm(
        log10B, decay_mean, decay_sigma, norm_decay_gaussian
    )

    # Central sloped region.
    central_region = np.where(
        (log10B >= rise_mean) & (log10B <= decay_mean),
        slope * (log10B - center) + 1,
        0,
    )

    # Combine the Gaussian rise, sloped region, and Gaussian decay
    smooth_top_hat = np.maximum(rise_gaussian, central_region)
    smooth_top_hat = np.maximum(smooth_top_hat, decay_gaussian)

    return smooth_top_hat


def initial_magnetic_field_lognormal(
    mean: float, sigma: float, NS_number: int
) -> np.ndarray:
    """
    Draw random initial magnetic field values from a log-normal distribution as suggested in Gullon et al. (2015).

    Args:
        mean (float): mean of the Gaussian initial period distribution, in [s].
        sigma (float): standard deviation of the initial period distribution, in [s].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    B_rand = 10 ** np.random.normal(mean, sigma, NS_number)

    return B_rand


def initial_magnetic_field_double_lognormal(NS_number: int) -> np.ndarray:
    """
    Draw random initial magnetic field values from a double log-normal distribution.

    Args:
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    log10B_grid = np.linspace(
        cfg["B_initial_log10_min"],
        cfg["B_initial_log10_max"],
        cfg["resolution"],
    )
    B_rand = 10 ** rs.random_from_pdf(
        log10B_grid, pdf_log10_magnetic_field_2normal, NS_number
    )

    return B_rand


def initial_magnetic_field_smooth_tophat(NS_number: int) -> np.ndarray:
    """
    Draw random initial magnetic field values from a smooth top-hat distribution.

    Args:
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    log10B_grid = np.linspace(
        cfg["B_initial_log10_min"],
        cfg["B_initial_log10_max"],
        cfg["resolution"],
    )
    B_rand = 10 ** rs.random_from_pdf(
        log10B_grid, pdf_log10_magnetic_field_smooth_tophat, NS_number
    )

    return B_rand
