"""
Conversions between different coordinate systems and related issues.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Tuple

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import galactocentric_frame_defaults


def check_radial_coordinate(r: float) -> None:
    """
    Check that the distance from the origin is not negative.

    Args:
        r (float): distance from the origin in units of length

    Returns:
        Returns None if r greater than or equal to 0,
        otherwise raises ValueError.
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
    Calculating the Cartesian x, y and z coordinates from spherical coordinates
    r, theta and psi.

    Args:
        r (float): radial component (magnitude of the vector) in spherical
        coordinates, r>0
        theta (float): polar angle, [0, pi]
        psi (float): azimuthal angle, [0, 2*pi]

    Returns:
        (float, float, float): x, y and z coordinates in a Cartesian system
    """

    x = r * np.sin(theta) * np.cos(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(theta)

    return x, y, z


def speed_cylindrical_to_cartesian(
    v_r: float, v_phi: float, v_z: float, phi: float
) -> Tuple[float, float, float]:
    """
        Calculating the galactocentric Cartesian v_x, v_y and v_z velocity components
        from cylindrical galactocentric components v_r, v_phi and v_z.

        Args:
            v_r (float): radial velocity component in a cylindrical galactocentric frame
            v_phi (float): azimuthal velocity component in a cylindrical
            galactocentric frame
            v_z (float): z velocity component in a cylindrical galactocentric frame
        Returns:
            (float, float, float): v_x, v_y and v_z velocity components in a Cartesian
            galactocentric frame.
        """

    v_x = v_r * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(phi) + v_phi * np.cos(phi)

    return v_x, v_y, v_z


def galactocentric_to_icrs(
    x: float, y: float, z: float, v_x: float, v_y: float, v_z: float
) -> Tuple[float, float, float, float]:
    """
        Calculating the ICRS (International Celestial Reference Frame) coordinates RA,
        DEC and proper velocities v_RA, v_DEC from galactocentric spatial coordinates
        and velocities x, y, z, v_x, v_y and v_z. This galactocentric coordinates
        refers to the galactocentric reference frame used in the simulation defined
        as a right-handed reference frame with the Sun located at the coordinate
        point (x = 0 kpc, y = 8.5 kpc, z = 0.02 kpc).
        We use the astropy.coordinates package that allows automatic conversions
        between coordinate systems.

        Args:
            x (float): x coordinate in kpc in galactocentric reference frame
            y (float): y coordinate in kpc in galactocentric reference frame
            z (float): z coordinate in kpc in galactocentric reference frame
            v_x (float): x velocity component in km/s in galactocentric reference frame
            v_y (float): y velocity component in km/s in galactocentric reference frame
            v_z (float): z velocity component in km/s in galactocentric reference frame

        Returns:
            (float, float, float, float): RA, DEC coordinates in degree and v_RA
            v_DEC proper velocity components in mas/yr in the ICRS reference frame.
        """

    # set the astropy galactocentric frame with the most recent updated parameter
    # values from latest astropy version
    _ = galactocentric_frame_defaults.set("v4.0")

    # The galactocentric reference frame used in the simulation is a right-handed
    # reference frame with the Sun located at the coordinate point (x = 0 kpc,
    # y = 8.5 kpc, z = 0.02 kpc).
    # The module astropy.coordinates.Galactocentric deals with galactocentric
    # coordinates but it is defined with the x, y, axes rotated of 90 degrees
    # clockwise respect to the galactocentric reference frame used in the simulation.
    # In this new frame the position of the Sun is (x = -8.5 kpc, y = 0 kpc, z = 0.02
    # kpc). We therefore need to convert the galactocentric coordinates we used in
    # the simulation into the galactocentric frame defined in astropy. To do that we
    # apply the transformation (x -> y_gal, y -> -x_gal, z -> z_gal).

    x_gal = -y
    y_gal = x
    z_gal = z

    v_x_gal = -v_y
    v_y_gal = v_x
    v_z_gal = v_z

    # create an object containing the coordinates using the class
    # coordinates.Galactocentric from astropy
    gc_coord = coord.Galactocentric(
        x=x_gal * u.kpc,
        y=y_gal * u.kpc,
        z=z_gal * u.kpc,
        v_x=v_x_gal * (u.km / u.s),
        v_y=v_y_gal * (u.km / u.s),
        v_z=v_z_gal * (u.km / u.s),
        z_sun=0.02 * u.kpc,
        galcen_distance=8.5 * u.kpc,
    )

    # transform from galactocentric to ICRS frame
    icrs_coord = gc_coord.transform_to(coord.ICRS)

    # convert RA and DEC units in degrees in the ranges [0, 360] deg and [-90,
    # 90] deg respectively and remove astropy units to obtain numpy float values.
    ra = icrs_coord.ra.degree / u.deg
    dec = icrs_coord.dec.degree / u.deg
    v_ra = icrs_coord.pm_ra_cosdec / (u.mas / u.yr)
    v_dec = icrs_coord.pm_dec / (u.mas / u.yr)

    return ra, dec, v_ra, v_dec
