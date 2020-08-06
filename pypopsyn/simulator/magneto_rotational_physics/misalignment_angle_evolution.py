"""
Evolution of the pulsar misalignment angle.

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

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.configuration import cfg


def misalignment_angle_derivative(
    t: float, chi: float, B: float, P: float
) -> float:
    """
    This function determines the change in the misalignment angle, i.e., the angle between the
    magnetic dipolar moment and the rotation axis of a pulsar. It is taken from eq. (71) of
    Pons & Vigano (2019). For more details see, e.g., Spitkovsky (2006) or Philippov et al.
    (2014), who determine the coefficients k_0, k_1, k_2 (defined in the configuration file)
    for a pulsar embedded in a force-free and resistive magnetosphere from numerical simulations.
    Note that all three input parameters are time-dependent.

    Args:
        t (float): unused current time parameter in [yr], only needed for integration below.
        chi (float): angles between the magnetic dipolar moment, i.e., the magnetic
        field axis, and the rotation axis for all simulated pulsars, measured in [rad].
        B (float): values of the dipolar component of the magnetic field at the
        magnetic pole for the sample of simulated neutron stars, measured in [G].
        P (float): spin periods of simulated pulsars, measured in [s].

    Returns:
        (float): misalignment angle derivatives for all simulated pulsars in [rad/s].
    """

    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect solid sphere.
    NS_inertia = 2.0 / 5.0 * cfg["NS_mass"] * cfg["NS_radius"] ** 2

    # Auxiliary quantity beta as defined in eq. (72) of Pons & Vigano (2019).
    beta = np.pi ** 2 * cfg["NS_radius"] ** 6 / (NS_inertia * const.c ** 3)

    # Misalignment angle derivative.
    chi_deriv = (
        -cfg["k_coefficients"][2]
        * beta
        * B ** 2
        / (P ** 2)
        * np.sin(chi)
        * np.cos(chi)
    )

    return chi_deriv
