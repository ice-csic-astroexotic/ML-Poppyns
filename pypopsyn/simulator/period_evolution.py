"""
Evolution of the pulsar period

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Tuple

import numpy as np

import pypopsyn.simulator.constants as const
import pypopsyn.simulator.coordinate_conversions as coco
from pypopsyn.simulator.configuration import cfg


def period_derivative(
    B: np.ndarray, chi: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    This function determines the change in the rotation period of pulsars embedded in
    a force-free and resistive magnetosphere. It is taken from Eqn (70) of Pons & Vigano
    (2019). For more details see, e.g., Spitkovsky (2006) or Philippov et al. (2014),
    which determine the coefficients k_0, k_1, k_2 from numerical simulations.

    Args:
        B (np.ndarray): values of the dipolar component of the magnetic field at the
        magnetic pole for the sample of simulated neutron stars, measured in [G].
        P (np.ndarray): spin periods of simulated pulsars, measured in [s].
        chi (np.ndarray): angles between the magnetic dipolar moment, i.e., the magnetic
        field axis, and the rotation axis for all simulated pulsars, measured in [rad].

    Returns:
        (np.ndarray): period derivatives for all simulated pulsars in [s/s].

    """

    # Dimensionless coefficients k_0, k_1, k_2
    k = np.array([1.0, 1.0, 1.0])

    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect sphere
    NS_inertia = 2.0 / 5.0 * cfg["NS_mass"] * cfg["NS_radius"] ** 2

    # Auxiliary quantity beta as defined in Eqn (72) of Pons & Vigano (2019)
    beta = np.pi ** 2 * cfg["mass"] ** 6 / (NS_inertia * const.c ** 3)

    P_deriv = beta * B ** 2 / P * (k[0] + k[1] * np.sin(chi) ** 2)

    return P_deriv
