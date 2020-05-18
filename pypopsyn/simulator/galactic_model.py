"""
Model for the Milky Way gravitational potential

We consider the same galactic structure as in Faucher-Giguère & Kaspi (2006). Their
model consists of three components: a disk-halo, a bulge and a nucleus.
The parameters of the model are taken from table B1 in Kuijken & Gilmore (1989).

To improve performance when evolving the neutron stars' position in the galactic
potential (see dynamical_evolution.py), we add Numba's jit decorator to all functions.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Tuple

import numpy as np
from numba import jit

import pypopsyn.simulator.constants as const


@jit
def shape_parameter(z: float) -> Tuple[float, float]:
    """
    Shape parameter for the disk-halo potential from Carlberg & Innamen (1987) and
    its derivative with respect to the height z from the galactic disk.
    First term in the denominator of eq. (13) in Faucher-Giguère & Kaspi (2006).

    Args:
        z (float): height from the galactic disk in kpc

    Returns:
        (float, float): value of the shape parameter and its derivative with respect
        to z
    """
    # parameter values from Faucher-Giguère & Kaspi (2006), Kuijken & Gilmore (1989)
    a_d = 2.4  # scale length of the disk in kpc
    h = np.array(
        [0.325, 0.090, 0.125]
    )  # array of disk components' scale heights in kpc
    beta = np.array(
        [0.4, 0.5, 0.1]
    )  # array of weights for the disk components

    K = (
        a_d
        + beta[0] * np.sqrt(z ** 2 + h[0] ** 2)
        + beta[1] * np.sqrt(z ** 2 + h[1] ** 2)
        + beta[2] * np.sqrt(z ** 2 + h[2] ** 2)
    )

    dK_dz = (
        beta[0] * z / np.sqrt(z ** 2 + h[0] ** 2)
        + beta[1] * z / np.sqrt(z ** 2 + h[1] ** 2)
        + beta[2] * z / np.sqrt(z ** 2 + h[2] ** 2)
    )

    return K, dK_dz


@jit
def r_z_derivatives_dh_potential(r: float, z: float) -> Tuple[float, float]:
    """
    Derivative with respect to r and z of the disk-halo component gravitational
    potential defined in eq. (14) in Faucher-Giguère & Kaspi (2006).

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc
        z (float): height from the galactic disk in kpc

    Returns:
        (float, float): derivative with respect to r and z of the disk-halo potential
    """
    # parameters for the disk-halo potential
    # (Faucher-Giguère & Kaspi 2006, Kuijken & Gilmore 1989)
    M_dh = 1.45e11 * const.M_SUN  # disk+halo mass in g
    b_dh = 5.5  # core radius of the halo component in kpc

    K, dK_dz = shape_parameter(z)
    dpot_dh_dr = (
        const.G_KPC_YR
        * M_dh
        * r
        * (K ** 2 + b_dh ** 2 + r ** 2) ** (-3.0 / 2.0)
    )
    dpot_dh_dz = (
        const.G_KPC_YR
        * M_dh
        * (K ** 2 + b_dh ** 2 + r ** 2) ** (-3.0 / 2.0)
        * K
        * dK_dz
    )

    return dpot_dh_dr, dpot_dh_dz


@jit
def r_derivative_b_potential(r: float) -> float:
    """
    Derivative with respect to r of the bulge component gravitational potential
    defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc

    Returns:
        float: derivative with respect to r of the bulge potential
    """
    # parameters for the bulge potential
    # (Faucher-Giguère & Kaspi 2006, Kuijken & Gilmore 1989)
    M_b = 9.3e9 * const.M_SUN  # bulge mass in g
    b_b = 0.25  # core radius of the bulge component in kpc
    dpot_b_dr = const.G_KPC_YR * M_b * r * (b_b ** 2 + r ** 2) ** (-3.0 / 2.0)

    return dpot_b_dr


@jit
def r_derivative_n_potential(r: float) -> float:
    """
    Derivative with respect to r of the nucleus component gravitational potential
    defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc

    Returns:
        float: derivative with respect to r of the nucleus potential
    """
    # parameters for the nucleus potential
    # (Faucher-Giguère & Kaspi 2006, Kuijken & Gilmore 1989)
    M_n = 1.0e10 * const.M_SUN  # nucleus mass in g
    b_n = 1.5  # core radius of the nucleus component in kpc

    dpot_n_dr = const.G_KPC_YR * M_n * r * (b_n ** 2 + r ** 2) ** (-3.0 / 2.0)

    return dpot_n_dr


@jit
def cylind_coord_gradient_mw_potential(r: float, z: float) -> np.ndarray:
    """
    Gradient in cylindrical coordinates of the Milky Way gravitational potential
    defined in eq. (13) in Faucher-Giguère & Kaspi (2006).

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc
        z (float): height from the galactic disk in kpc

    Returns:
        (np.ndarray): gradient of the galactic potential in cylindrical
        coordinates
    """

    dpot_dh_dr, dpot_dh_dz = r_z_derivatives_dh_potential(r, z)
    dpot_b_dr = r_derivative_b_potential(r)
    dpot_n_dr = r_derivative_n_potential(r)

    dpot_mw_dr = dpot_dh_dr + dpot_b_dr + dpot_n_dr
    dpot_mw_dphi = 0.0
    dpot_mw_dz = dpot_dh_dz

    pot_mw_gradient = np.array([dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz])

    return pot_mw_gradient
