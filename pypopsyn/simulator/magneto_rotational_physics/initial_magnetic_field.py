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
        (np.ndarray): value of the pdf for each B.
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


def initial_magnetic_field_double_lognormal(
    log10B_min: float, log10B_max: float, NS_number: int
) -> np.ndarray:
    """
    Double log-normal distributed initial magnetic field as suggested.
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
