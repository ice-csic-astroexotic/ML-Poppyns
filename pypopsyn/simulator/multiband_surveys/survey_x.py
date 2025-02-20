"""
    Model for the pulsar X-ray surveys.

    We consider here the detection of thermally emitting neutron stars.


    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np


def detected_x_population_flux_threshold(
    S_x_abs: np.ndarray,
    S_x_abs_threshold: float,
) -> np.ndarray:
    """
    Compute the pulsars detected by an X-ray survey with a given flux threshold.

    Args:
        S_x_abs (np.ndarray): Array of X-ray absorbed fluxes in [erg s^-1 cm^-2]
        S_x_abs_threshold (float): The absorbed flux threshold for X-ray detection in [erg s^-1 cm^-2].

    Returns:
        (np.ndarray): Boolean mask to select the pulsars detected above a given flux threshold.
    """

    # Filter the neutron stars according to a threshold flux.
    detected_x = S_x_abs > S_x_abs_threshold

    return detected_x
