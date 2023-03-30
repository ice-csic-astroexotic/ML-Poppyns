"""
Initial probability distribution of pulsar periods.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

MIT License

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

import numpy as np

import pypopsyn.simulator.basics.random_sampler as rs
from pypopsyn.simulator.configuration import cfg


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


def pdf_period_powerlaw(
    P: np.ndarray,
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

    power_law_index = -1
    pdf = P**power_law_index

    # pdf = 1/P * np.exp(-0.5*(np.log(P) - 2.)**2)

    return pdf


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

    P_lognorm = 10 ** np.random.normal(mean, sigma, int(0.5 * NS_number))

    P_grid = np.logspace(mean, 4, cfg["resolution"])
    P_powerlaw = rs.random_from_pdf(
        P_grid, pdf_period_powerlaw, int(0.5 * NS_number)
    )

    P_initial = np.concatenate([P_lognorm, P_powerlaw])

    while len(P_initial) < NS_number:
        P_initial = np.append(
            P_initial, rs.random_from_pdf(P_grid, pdf_period_powerlaw, 1)
        )

    return P_initial
