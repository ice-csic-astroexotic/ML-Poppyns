"""
Initial velocity distribution for the stellar population.

The velocity is composed of two contributions, the neutron stars' proper motion
caused by kicks during the supernova as well as the motion of the galaxy itself.
For the former, we follow Gullon et al. (2014).
"""

import numpy as np

import pypopsyn.simulator.galactic_model as gm


def pdf_proper_velocity(v: float) -> float:
    """
    Probability density function for the neutron stars' proper velocities.

    Args:
        v (float): proper velocity in km/s

    Returns:
        float: stellar proper velocity distribution in 1/(km/s)
    """

    # we follow Gullon et al. (2014) and consider an exponential distribution

    v_mean = 600.0  # [km/s]
    v_p = 1.0 / v_mean * np.exp(-v / v_mean)

    return v_p


def virial_orbital_velocity(r: float, z: float) -> float:
    """
    Orbital virial velocity in kpc / yr for a circular orbit at a distance r from the galactic
    center and at an height z from the galactic plain. This velocity is evaluated by
    assuming equilibrium between the gravitational acceleration in the r direction
    due to the galactic potential and the centrifugal acceleration due to rotation.

    Args:
        r (float): distance in the galactic disk from the galactic centre in kpc
        z (float): height from the galactic disk in kpc

    Returns:
        (float): value of the orbital virial velocity in kpc / yr
    """
    pot_mw_gradient = gm.cylind_coord_gradient_mw_potential(r, z)
    v_virial = np.sqrt(r * pot_mw_gradient[0])

    return v_virial
