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

from typing import Tuple

import numpy as np


def pdf_period(mean: float, NS_number: int, sigma: float) -> np.ndarray:
    """
    We follow Faucher-Giguère & Kaspi (2006) and Gullon et al. (2014) and assume
    that the spin periods follow a normal (Gaussian) distribution with a certain
    mean and standard deviation that are defined in the configuration file. Note
    that for physical reasons, we reject negative spin-periods and redraw them
    again from the Gaussian distribution.

    Args:
        mean (float): mean of the Gaussian initial period distribution, in [s].
        NS_number (int): total number of neutron stars created in the simulation.
        sigma (float): standard deviation of the initial period distribution, in [s].

    Returns:
        (np.ndarray): initial pulsar period in [s] drawn from a Gaussian distribution.
    """

    P_initial = np.zeros(NS_number)

    for i in range(NS_number):
        P_initial[i] = np.random.normal(mean, sigma, 1)
        while P_initial[i] <= 0:
            P_initial[i] = np.random.normal(mean, sigma, 1)

    return P_initial
