"""
    Model for the pulsar X-ray surveys.

    We consider here the detection of thermally emitting neutron stars.


    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np


def detected_x_population_sharp_flux_filter(
    S_x: np.ndarray,
    S_x_threshold: float = 1.0e-15,
) -> np.ndarray:
    """
    Compute the pulsars detected by an X-ray survey with a given sharp flux threshold.

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold (float): The flux threshold for X-ray detection in [erg s^-1 cm^-2].

    Returns:
        (np.ndarray): Boolean mask to select the pulsars detected above a given flux threshold.
    """

    # Filter the neutron stars according to a threshold flux.
    detected_x = S_x > S_x_threshold

    return detected_x


def detected_x_population_smooth_flux_filter(
    S_x: np.ndarray,
    S_x_threshold_log10_mean: float = -15,
    S_x_threshold_log10_sigma: float = 0.5,
) -> np.ndarray:
    """
    Compute the pulsars detected by an X-ray survey with a given flux threshold from a gaussian distribution in log10
    to mimic all the uncertainties inherent in a detection with an instrument.

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold_log10_mean (float): The mean of the flux threshold distribution for X-ray detection in [erg s^-1 cm^-2].
        S_x_threshold_log10_sigma (float): The standard deviation of the flux threshold distribution for X-ray detection in
            [erg s^-1 cm^-2].

    Returns:
        (np.ndarray): Boolean mask to select the pulsars detected above a given flux threshold.
    """
    flux_threshold = 10 ** np.random.normal(
        S_x_threshold_log10_mean,
        S_x_threshold_log10_sigma,
        len(S_x),
    )
    detected_mask = S_x > flux_threshold
    return detected_mask
