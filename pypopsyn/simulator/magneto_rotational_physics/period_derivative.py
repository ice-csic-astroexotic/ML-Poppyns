"""
    Time derivative of the pulsar period.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
from numba import float64, jit

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.config_simulator import cfg

# Redefining global variables to allow type specification.
# Necessary right now in order to get JIT to work.
NS_mass: float = cfg["NS_mass"]
NS_radius: float = cfg["NS_radius"]
k_coefficients_0: float = cfg["k_coefficients"][0]
k_coefficients_1: float = cfg["k_coefficients"][1]


@jit(float64(float64, float64, float64))
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
        (float): period derivative of a simulated pulsar in [s/yr].
    """

    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect solid sphere.
    NS_inertia = 2.0 / 5.0 * NS_mass * NS_radius**2

    # Auxiliary quantity beta as defined in eq. (72) of Pons & Vigano (2019).
    beta = np.pi**2 * NS_radius**6 / (NS_inertia * const.C**3)

    # Period derivative.
    P_deriv = (
        beta
        * B**2
        / P
        * (k_coefficients_0 + k_coefficients_1 * np.sin(chi) ** 2)
    ) * const.YR_TO_S

    return P_deriv
