"""
    Initial probability distribution of pulsar periods.

    Authors:

            Vanessa Graber (graber@ice.csic.es)
            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np


def pdf_period_normal(mean: float, sigma: float, NS_number: int) -> np.ndarray:
    """
    Normal (Gaussian) distribution for the initial spin periods as suggested in
    Faucher-Giguère & Kaspi (2006) and Gullon et al. (2014).
    The mean and standard deviation are defined in the configuration file. Note
    that for physical reasons, we reject negative spin-periods and redraw them
    again from the Gaussian distribution.

    Args:
        mean (float): mean of the Gaussian initial period distribution, in [s].
        sigma (float): standard deviation of the initial period distribution, in [s].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial pulsar period in [s] drawn from a Gaussian distribution.
    """

    P_initial = np.zeros(NS_number)

    for i in range(NS_number):
        P_initial[i] = np.random.normal(mean, sigma, 1)
        while P_initial[i] <= 0:
            P_initial[i] = np.random.normal(mean, sigma, 1)

    return P_initial


def pdf_period_lognormal(
    mean: float, sigma: float, NS_number: int
) -> np.ndarray:
    """
    Log-normal distribution for the initial spin periods as suggested in Igoshev et al. (2022).
    The mean and standard deviation are defined in the configuration file.

    Args:
        mean (float): mean of the Gaussian initial period distribution, in [s].
        sigma (float): standard deviation of the initial period distribution, in [s].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): initial pulsar period in [s] drawn from a Log-normal distribution.
    """

    P_initial = 10 ** np.random.normal(mean, sigma, NS_number)

    return P_initial
