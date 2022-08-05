"""
Initial velocity distribution for the stellar population.

This implementation relies on solving the ODEs using julia.

The velocity is composed of two contributions, the neutron stars' proper motion
caused by kicks during the supernova as well as the motion of the galaxy itself.
For the former, we follow Gullon et al. (2014).

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Borja Miñano (borja.minano @ uib.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)


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
from julia import Main

from pypopsyn.simulator.configuration import cfg


def pdf_kick_velocity_exp(v: np.ndarray) -> np.ndarray:
    """
    Spherically symmetric exponential probability density function for the neutron stars' initial
    kick velocity magnitude following eq. (5) in Ofek (2009).

    Args:
        v (np.ndarray): initial kick velocity magnitude in [km/s].

    Returns:
        np.ndarray: stellar kick velocity distribution in [1/(km/s)].
    """
    vk_mean = cfg["vk_c"]
    pdf_vk = v / vk_mean**2 * np.exp(-v / vk_mean)

    return pdf_vk


def pdf_kick_velocity_maxwell(v: np.ndarray) -> np.ndarray:
    """
    Maxwell probability density function for the neutron stars' initial kick
    velocity magnitude following section 6.2 in Hobbs et al. (2005).

    Args:
        v (np.ndarray): initial kick velocity magnitude in [km/s].

    Returns:
        np.ndarray: stellar kick velocity distribution in [1/(km/s)].
    """
    sigma = cfg["sigma_k"]
    pdf_vk = (
        np.sqrt(2 / np.pi)
        * v**2
        / (sigma**3)
        * np.exp(-(v**2) / (2 * sigma**2))
    )

    return pdf_vk


def pdf_kick_velocity_2maxwell(v: np.ndarray) -> np.ndarray:
    """
    Double Maxwell probability density function for the neutron stars' initial kick
    velocity magnitude following eq. (5) and section 4.2 in Igoshev (2020).

    Args:
        v (np.ndarray): initial kick velocity magnitude in [km/s].

    Returns:
        np.ndarray: stellar kick velocity distribution in [1/(km/s)].
    """

    # Define the dispersions of the two Maxwellian components.
    sigma_1 = 55
    sigma_2 = 334
    # Define the fractional contribution of the first Maxwellian.
    w = 0.19

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
        r (float): distance in the galactic disk from the galactic center in [kpc].
        z (float): height from the galactic disk in [kpc].

    Returns:
        (float): value of the circular velocity in [kpc/yr].
    """

    # Setting names in the ´Main´ module to send Python values to Julia.
    Main.r = r
    Main.z = z
    Main.galactic_model_input = cfg["galactic_model"]

    # Importing the ´galactic_model.jl´ file with Julia code into our Main Julia.
    Main.include("pypopsyn/simulator_julia/stellar_dynamics/galactic_model.jl")

    # Evaluating cylind_coord_gradient_mw_potential function in Julia.
    pot_mw_gradient = Main.eval(
        "cylind_coord_gradient_mw_potential(galactic_model, r, z)"
    )
    v_circular = np.sqrt(r * pot_mw_gradient[0])

    return v_circular
