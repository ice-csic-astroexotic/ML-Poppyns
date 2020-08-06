"""
Time derivative of the pulsar period.

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
from numba import jit

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.configuration import cfg


# @jit
def period_derivative(B: float, chi: float, P: float) -> float:
    """
    This function determines the change in the rotation period of a pulsar. It is taken
    from eq. (70) of Pons & Vigano (2019). For more details see, e.g., Spitkovsky (2006)
    or Philippov et al. (2014), who determine the coefficients k_0, k_1, k_2 (defined in
    the configuration file) for a pulsar embedded in a force-free and resistive magnetosphere
    from numerical simulations. Note that all three input parameters are time-dependent.

    Args:
        B (float): value of the dipolar component of the magnetic field at the
        magnetic pole for a simulated neutron star, measured in [G].
        chi (float): angle between the magnetic dipolar moment, i.e., the magnetic
        field axis, and the rotation axis for a simulated pulsar, measured in [rad].
        P (float): spin period of a simulated pulsar, measured in [s].

    Returns:
        (float): period derivative of a simulated pulsar in [s/s].
    """

    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect solid sphere.
    NS_inertia = 2.0 / 5.0 * cfg["NS_mass"] * cfg["NS_radius"] ** 2

    # Auxiliary quantity beta as defined in eq. (72) of Pons & Vigano (2019).
    beta = np.pi ** 2 * cfg["NS_radius"] ** 6 / (NS_inertia * const.c ** 3)

    # Period derivative.
    P_deriv = (
        beta
        * B ** 2
        / P
        * (
            cfg["k_coefficients"][0]
            + cfg["k_coefficients"][1] * np.sin(chi) ** 2
        )
    )

    return P_deriv
