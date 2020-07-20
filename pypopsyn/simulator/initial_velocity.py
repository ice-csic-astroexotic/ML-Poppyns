"""
Initial velocity distribution for the stellar population

The velocity is composed of two contributions, the neutron stars' proper motion
caused by kicks during the supernova as well as the motion of the galaxy itself.
For the former, we follow Gullon et al. (2014).

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np

import pypopsyn.simulator.galactic_model as gm
from pypopsyn.simulator.configuration import cfg

galactic_model = cfg["galactic_model"]
if galactic_model == "gmM19":
    gmod = gm.GalaxyModelM19()
elif galactic_model == "gmFK06":
    gmod = gm.GalaxyModelFK06()
else:
    raise ValueError(
        "The galactic model does not exist. Choose between gmFK06 or gmM19."
    )


def pdf_kick_velocity_exp(v: float) -> float:
    """
    Decaying exponential Probability density function for the neutron stars' initial kick
    velocity magnitude following eq. (3) in Gullon et al. (2014).

    Args:
        v (float): initial kick velocity magnitude in km / s.

    Returns:
        float: stellar kick velocity distribution in 1 / (km / s).
    """
    vk_mean = cfg["vk_c"]
    pdf_vk = 1.0 / vk_mean * np.exp(-v / vk_mean)

    return pdf_vk


def pdf_kick_velocity_maxwell(v: float) -> float:
    """
    Maxwell probability density function for the neutron stars' initial kick
    velocity magnitude following eq. (3) in Gullon et al. (2014).

    Args:
        v (float): initial kick velocity magnitude in km / s.

    Returns:
        float: stellar kick velocity distribution in 1 / (km / s).
    """
    sigma = cfg["vk_c"]
    pdf_vk = v ** 2 * np.exp(-(v ** 2) / (2 * sigma ** 2))

    return pdf_vk


def circular_velocity(r: float, z: float) -> float:
    """
    Circular velocity in kpc / yr for a circular orbit at a distance r from the
    galactic center and at a height z from the galactic plane. This velocity is
    evaluated by assuming equilibrium between the gravitational acceleration in the
    r direction due to the galactic potential and the centrifugal acceleration due
    to rotation.

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc.
        z (float): height from the galactic disk in kpc.

    Returns:
        (float): value of the circular velocity in kpc / yr.
    """

    pot_mw_gradient = gmod.cylind_coord_gradient_mw_potential(r, z)
    v_circular = np.sqrt(r * pot_mw_gradient[0])

    return v_circular
