"""
Conversions between different coordinate systems and related issues
"""

from typing import Tuple

import numpy as np


def check_radial_coordinate(r: float):
    """
    Check that the distance from the origin is not negative.

    Args:
        r (float): distance from the origin in units of length
    """
    if r < 0:
        raise ValueError("Radial coordinate is out of range")


def polar_to_cartesian(r: float, phi: float) -> Tuple[float, float]:
    """
    Calculating the Cartesian x and y coordinates from plane polar r and phi.

    Args:
        r (float): radial component (magnitude of the vector) in plane polar
        coordinates, r>0
        phi (float): angular coordinate, [0, 2*pi]

    Returns:
        (float, float): x and y coordinates in a Cartesian system
    """

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def spherical_to_cartesian(
    r: float, theta: float, psi: float
) -> Tuple[float, float, float]:
    """
    Calculating the Cartesian x, y and z coordinates from spherical coordinates r, theta and psi.

    Args:
        r (float): radial component (magnitude of the vector) in spherical
        coordinates, r>0
        theta (float): polar angle, [0, pi]
        psi (float): azimuthal angle, [0, 2*pi]

    Returns:
        (float, float): x, y and z coordinates in a Cartesian System
    """

    x = r * np.sin(theta) * np.cos(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(theta)

    return x, y, z
