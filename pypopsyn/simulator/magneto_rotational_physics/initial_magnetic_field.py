"""
    Initial probability distribution of pulsar periods.

    Authors:

            Vanessa Graber (graber@ice.csic.es)
            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np

import utilities.samplers.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg


def initial_magnetic_field_lognormal(
    mean: float, sigma: float, NS_number: int
) -> np.ndarray:
    """
    Log-normal distribution for the initial magnetic field as suggested in Gullon et al. (2015).
    The mean and standard deviation are defined in the configuration file.

    Args:
        mean (float): mean of the Gaussian initial period distribution, in [s].
        sigma (float): standard deviation of the initial period distribution, in [s].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    B_rand = 10 ** np.random.normal(mean, sigma, NS_number)

    return B_rand


def pdf_log10_magnetic_field_2normal(log10B: np.ndarray) -> np.ndarray:
    """
    Double Normal probability density function for the neutron stars' initial magnetic field.

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


def gaussian(x: np.ndarray, mean: float, sigma: float, norm: float):
    """
    A Gaussian function with custom normalization.

    Args:
        x (np.ndarray): array of values where to compute the function.
        mean (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        norm (float): Normalization of the Gaussian distribution.
    """
    gauss_func = norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    return gauss_func


def pdf_log10_magnetic_field_smooth_tophat(
    log10B: np.ndarray,
) -> np.ndarray:
    """
    A smooth top-hat function with a sloped central region and gaussian rise and decay.

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

    center = (rise_mean + decay_mean) / 2

    norm_rise_gaussian = slope * (rise_mean - center) + 1
    norm_decay_gaussian = slope * (decay_mean - center) + 1

    # Gaussian rise on the left side
    rise_gaussian = gaussian(log10B, rise_mean, rise_sigma, norm_rise_gaussian)

    # Gaussian decay on the right side
    decay_gaussian = gaussian(
        log10B, decay_mean, decay_sigma, norm_decay_gaussian
    )

    # Central sloped region.
    central_region = np.where(
        (log10B >= rise_mean) & (log10B <= decay_mean),
        slope * (log10B - center) + 1,
        0,
    )

    # Combine the Gaussian rise, flat region, and Gaussian decay
    smooth_top_hat = np.maximum(rise_gaussian, central_region)
    smooth_top_hat = np.maximum(smooth_top_hat, decay_gaussian)

    return smooth_top_hat


def initial_magnetic_field_double_lognormal(
    log10B_min: float, log10B_max: float, NS_number: int
) -> np.ndarray:
    """
    Double log-normal distributed initial magnetic field.
    The mean and standard deviation are defined in the configuration file.

    Args:
        log10B_min (float): minimum value of the log10 of the initial magnetic field in [G].
        sigma (float): maximum value of the log10 of the initial magnetic field in [G].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    log10B_grid = np.linspace(log10B_min, log10B_max, cfg["resolution"])
    B_rand = 10 ** rs.random_from_pdf(
        log10B_grid, pdf_log10_magnetic_field_2normal, NS_number
    )

    return B_rand


def initial_magnetic_field_smooth_tophat(
    log10B_min: float, log10B_max: float, NS_number: int
) -> np.ndarray:
    """
    Smooth top-hat distributed initial magnetic field.
    The parameters are defined in the configuration file.

    Args:
        log10B_min (float): minimum value of the log10 of the initial magnetic field in [G].
        sigma (float): maximum value of the log10 of the initial magnetic field in [G].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial magnetic field in [G].
    """

    log10B_grid = np.linspace(log10B_min, log10B_max, cfg["resolution"])
    B_rand = 10 ** rs.random_from_pdf(
        log10B_grid, pdf_log10_magnetic_field_smooth_tophat, NS_number
    )

    return B_rand
