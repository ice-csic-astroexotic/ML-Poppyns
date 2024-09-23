"""
    Model for the pulsar x-ray surveys.

    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np


def obs_bias_filter(S_x: np.ndarray, eta: float) -> np.ndarray:
    """
    This method simulates the x-ray observational biases following Gullon et al. (2015).

    Args:
        S_x (np.ndarray): pulsars' x-ray absorbed fluxes in [erg s^-(1) cm^(-2)].
        eta: filter coefficient.

    Returns:
        (np.ndarray): array of boolean flags, True if a pulsar is detected in x-rays and False if it is not detected.
    """
    prob_rand = np.random.uniform(0, 1, len(S_x))

    p_detection = np.minimum(eta * (S_x / 1.0e-11), 1)

    mask_detected = prob_rand > (1 - p_detection)

    return mask_detected
