"""
    Initial velocity distribution for the stellar population.

    The velocity is composed of two contributions, the neutron stars' proper motion
    caused by kicks during the supernova as well as the motion of the galaxy itself.
    For the former, we follow Gullon et al. (2014).

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np

import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
from pypopsyn.simulator.config_simulator import cfg


def pdf_kick_velocity_exp(v: np.ndarray) -> np.ndarray:
    """
    Spherically symmetric exponential probability density function for the neutron stars' initial
    kick velocity magnitude following eq. (5) in Ofek (2009).

    Args:
        v (np.ndarray): Initial kick velocity magnitude in [km/s].

    Returns:
        (np.ndarray): Stellar kick velocity distribution in [1/(km/s)].
    """
    vk_mean = cfg["vk_c"]
    pdf_vk = v / vk_mean**2 * np.exp(-v / vk_mean)

    return pdf_vk


def pdf_kick_velocity_maxwell(v: np.ndarray) -> np.ndarray:
    """
    Maxwell probability density function for the neutron stars' initial kick
    velocity magnitude following section 6.2 in Hobbs et al. (2005).

    Args:
        v (np.ndarray): Initial kick velocity magnitude in [km/s].

    Returns:
        (np.ndarray): Stellar kick velocity distribution in [1/(km/s)].
    """
    sigma = cfg["sigma_k"]
    pdf_vk = (
        np.sqrt(2 / np.pi)
        * v**2
        / (sigma**3)
        * np.exp(-(v**2) / (2 * sigma**2))
    )

    return pdf_vk


def pdf_kick_velocity_double_maxwell(v: np.ndarray) -> np.ndarray:
    """
    Double Maxwell probability density function for the neutron stars' initial kick
    velocity magnitude following eq. (5) and section 4.2 in Igoshev (2020).

    Args:
        v (np.ndarray): Initial kick velocity magnitude in [km/s].

    Returns:
        (np.ndarray): Stellar kick velocity distribution in [1/(km/s)].
    """

    # Define the dispersions of the two Maxwellian components.
    sigma_1 = cfg["sigma_k_comp1"]
    sigma_2 = cfg["sigma_k_comp2"]
    # Define the fractional contribution of the first Maxwellian.
    w = cfg["kick_weight_comp1"]
    if (w < 0) or (w > 1):
        raise ValueError(
            "The relative weight parameter of the km_double_maxwell kick velocity model must be in "
            "the range 0 and 1."
        )

    pdf_maxwell_1 = (
        np.sqrt(2 / np.pi)
        * v**2
        / (sigma_1**3)
        * np.exp(-(v**2) / (2 * sigma_1**2))
    )

    pdf_maxwell_2 = (
        np.sqrt(2 / np.pi)
        * v**2
        / (sigma_2**3)
        * np.exp(-(v**2) / (2 * sigma_2**2))
    )

    pdf_vk = w * pdf_maxwell_1 + (1.0 - w) * pdf_maxwell_2

    return pdf_vk


def circular_velocity(r: float, z: float) -> float:
    """
    Circular velocity in [kpc/yr] for a circular orbit at a distance r from the
    galactic center and at a height z from the galactic plane. This velocity is
    evaluated by assuming equilibrium between the gravitational acceleration in the
    r direction due to the galactic potential and the centrifugal acceleration due
    to rotation.

    Args:
        r (float): Distance in the galactic disk from the galactic center in [kpc].
        z (float): Height from the galactic disk in [kpc].

    Returns:
        (float): Value of the circular velocity in [kpc/yr].
    """

    pot_mw_gradient = gm.galactic_model.cylind_coord_gradient_mw_potential(
        r, z
    )
    v_circular = np.sqrt(r * pot_mw_gradient[0])

    return v_circular
