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
elif galactic_model == "gmCI87":
    gmod = gm.GalaxyModelCI87()
else:
    raise ValueError("The galactic model does not exist")


def pdf_proper_velocity(v: float) -> float:
    """
    Probability density function for the neutron stars' initial proper
    velocity magnitude following eq. (3) in Gullon et al. (2014).

    Args:
        v (float): initial proper velocity magnitude in km / s.

    Returns:
        float: stellar proper velocity distribution in 1 / (km / s).
    """
    vp_mean = cfg["vp_mean"]
    pdf_vp = 1.0 / vp_mean * np.exp(-v / vp_mean)

    return pdf_vp


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
