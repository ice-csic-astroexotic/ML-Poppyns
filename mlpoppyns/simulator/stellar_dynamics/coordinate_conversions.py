"""
    Conversions between different coordinate systems and related issues.

    Authors:

            Vanessa Graber (graber@ice.csic.es)
            Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Tuple

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import galactocentric_frame_defaults

from mlpoppyns.simulator.config_simulator import cfg


def check_radial_coordinate(r: np.ndarray) -> None:
    """
    Check that the distance from the origin is not negative.

    Args:
        r (np.ndarray): Distance from the origin in units of length.
    """
    if np.any(r < 0):
        raise ValueError("One of the radial coordinates is out of range.")


def polar_to_cartesian(
    r: np.ndarray, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating the Cartesian x and y coordinates from plane polar r and phi.

    Args:
        r (np.ndarray): Radial component (magnitude of the vector) in plane polar coordinates, r>0.
        phi (np.ndarray): Angular coordinate, [0, 2*pi].

    Returns:
        (Tuple[np.ndarray, np.ndarray]): x and y coordinates in a Cartesian system.
    """

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def spherical_to_cartesian(
    r: np.ndarray, theta: np.ndarray, psi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculating the Cartesian x, y and z coordinates from spherical coordinates
    r, theta and psi.

    Args:
        r (np.ndarray): Radial component (magnitude of the vector) in spherical coordinates, r>0.
        theta (np.ndarray): Polar angle, [0, pi].
        psi (np.ndarray): Azimuthal angle, [0, 2*pi].

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): x, y and z coordinates in a Cartesian system.
    """

    x = r * np.sin(theta) * np.cos(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(theta)

    return x, y, z


def speed_cylindrical_to_cartesian(
    v_r: np.ndarray, v_phi: np.ndarray, v_z: np.ndarray, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculating the galactocentric Cartesian v_x, v_y and v_z velocity components
    from cylindrical galactocentric components v_r, v_phi and v_z.

    Args:
        v_r (np.ndarray): Radial velocity component in a cylindrical galactocentric frame.
        v_phi (np.ndarray): Azimuthal velocity component in a cylindrical galactocentric frame.
        v_z (np.ndarray): z velocity component in a cylindrical galactocentric frame.
        phi (np.ndarray): Azimuthal angle [0, 2*pi] in cylindrical coordinates.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): v_x, v_y and v_z velocity components in a Cartesian
            galactocentric frame.
    """

    v_x = v_r * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(phi) + v_phi * np.cos(phi)

    return v_x, v_y, v_z


def galactocentric_to_icrs(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Calculating the ICRS (International Celestial Reference Frame) coordinates RA, DEC, distance and proper velocities
    v_RA, v_DEC, v_ls from galactocentric spatial coordinates and velocities x, y, z, v_x, v_y and v_z. This
    galactocentric coordinates refers to the galactocentric reference frame used in the simulation defined as a
    right-handed reference frame with the Sun located at the coordinate point (x = 0 kpc, y = 8.5 kpc, z = 0.02 kpc).
    We use the astropy.coordinates package that allows automatic conversions between coordinate systems.

    Args:
        x (np.ndarray): x coordinate in [kpc] in galactocentric reference frame.
        y (np.ndarray): y coordinate in [kpc] in galactocentric reference frame.
        z (np.ndarray): z coordinate in [kpc] in galactocentric reference frame.
        v_x (np.ndarray): x velocity component in [km/s] in galactocentric reference frame.
        v_y (np.ndarray): y velocity component in [km/s] in galactocentric reference frame.
        v_z (np.ndarray): z velocity component in [km/s] in galactocentric reference frame.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): RA, DEC coordinates in [deg],
            distance from the ICRS origin in [kpc], proper motion pm_RA, pm_DEC components
            in [mas/yr] in the ICRS reference frame and the line of sight velocity in [km/s].
    """

    # Set the astropy galactocentric frame with the parameter
    # values from astropy version 4.0.
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

    # Create an object containing the coordinates using the class
    # coordinates.Galactocentric from astropy.
    gc_coord = coord.Galactocentric(
        x=x_gal * u.kpc,
        y=y_gal * u.kpc,
        z=z_gal * u.kpc,
        v_x=v_x_gal * (u.km / u.s),
        v_y=v_y_gal * (u.km / u.s),
        v_z=v_z_gal * (u.km / u.s),
        z_sun=cfg["z_sun"] * u.kpc,
        galcen_distance=cfg["R_sun"] * u.kpc,
    )

    # Transform from galactocentric to ICRS frame.
    icrs_coord = gc_coord.transform_to(coord.ICRS())

    # Determine RA and DEC in [deg] in the ranges [0, 360] and [-90, 90],
    # respectively, and drop the units to the astropy objects by taking only the values.
    ra = icrs_coord.ra.degree
    dec = icrs_coord.dec.degree
    sun_dist = icrs_coord.distance.value
    pm_ra = icrs_coord.pm_ra_cosdec.value
    pm_dec = icrs_coord.pm_dec.value
    v_ls = icrs_coord.radial_velocity.value

    return ra, dec, sun_dist, pm_ra, pm_dec, v_ls


def galactocentric_to_galactic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Calculating the galactic longitude l, galactic latitude b, distance, pm_l, pm_b proper motion components and line
    of sight velocity v_ls from the galactocentric coordinates x, y, z and velocity v_x, v_y, v_z.
    The galactocentric coordinates refer to the galactocentric reference frame defined as a right-handed reference
    frame with the Sun located at the coordinate point (x = 0 kpc, y = 8.3 kpc, z = 0.02 kpc).
    We use the astropy.coordinates package that allows automatic conversions between coordinate systems. l = 0 deg,
    b = 0 deg corresponds to the location of the Galactic center, l increase anticlockwise (in the direction of
    galactic rotation as seen from the Sun) and ranges in the interval [-180, 180] deg while b is in the range
    [-90, 90] deg.

    Args:
        x (np.ndarray): x coordinate in [kpc] in the galactocentric reference frame.
        y (np.ndarray): y coordinate in [kpc] in the galactocentric reference frame.
        z (np.ndarray): z coordinate in [kpc] in the galactocentric reference frame.
        v_x (np.ndarray): x component of the velocity in [km/s] in galactocentric reference frame.
        v_y (np.ndarray): y component of the velocity in [km/s] in galactocentric reference frame.
        v_z (np.ndarray): z component of the velocity in [km/s] in galactocentric reference frame.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Galactic longitude l,
            Galactic latitude b in [deg], distance in [kpc], pm_l, pm_b proper motion components in [mas/yr]
            and line of sight velocity v_ls in [km/s].
    """

    # The galactocentric reference frame used for the input is a right-handed
    # reference frame with the Sun located at the coordinate point (x = 0 kpc,
    # y = 8.3 kpc, z = 0.02 kpc).
    # The module astropy.coordinates.Galactocentric deals with galactocentric
    # coordinates but it is defined with the x, y, axes rotated by 90 degrees
    # clockwise with respect to the galactocentric reference frame used in the simulation.
    # In this new frame the position of the Sun is (x = -8.3 kpc, y = 0 kpc, z = 0.02
    # kpc). We therefore need to convert the galactocentric coordinates we used
    # into the galactocentric frame defined in astropy. To do that we
    # apply the transformation (x_gal -> y_astropy, y_gal -> -x_astropy, z_gal -> z_astropy).

    # Set the astropy galactocentric frame with the parameter
    # values from astropy version 4.0.
    _ = galactocentric_frame_defaults.set("v4.0")

    # Create an object containing the coordinates using the class
    # coordinates.Galactocentric from astropy.
    c = coord.Galactocentric(
        x=-y * u.kpc,
        y=x * u.kpc,
        z=z * u.kpc,
        v_x=-v_y * u.km / u.s,
        v_y=v_x * u.km / u.s,
        v_z=-v_z * u.km / u.s,
        z_sun=cfg["z_sun"] * u.kpc,
        galcen_distance=cfg["R_sun"] * u.kpc,
    )

    # Transform from galactocentric to galactic frame.
    galactic = c.transform_to(coord.Galactic())

    # Make the galactic longitude ranging in [-180, 180] deg with the galactic center at (0, 0) deg.
    l_gal = galactic.l.value
    l_gal[(l_gal > 180.0) & (l_gal <= 360.0)] = (
        l_gal[(l_gal > 180.0) & (l_gal <= 360.0)] - 360.0
    )
    b_gal = galactic.b.value
    distance = galactic.distance.value

    pm_l_cosb = galactic.pm_l_cosb.value
    pm_b = galactic.pm_b.value
    radial_velocity = galactic.radial_velocity.value

    return l_gal, b_gal, distance, pm_l_cosb, pm_b, radial_velocity


def convert_cylindrical_to_all_coordinates(dict_pop_dyn: dict) -> dict:
    """
    Converts a population's positions and velocities from galactocentric cylindrical
    (polar) coordinates to ICRS, and galactic coordinate frames.

    The resulting sky coordinates, distances, proper motions and line of sight velocity
    are added to the input dictionary.

    Args:
        dict_pop_dyn (dict): A dictionary containing the initial polar coordinates and velocities
            of a population of neutron stars.

    Returns:
        (dict): The input dictionary `dict_pop_dyn` with the following keys added:

            - 'ra' (np.ndarray): Right Ascension [deg].
            - 'dec' (np.ndarray): Declination [deg].
            - 'dist' (np.ndarray): Heliocentric distance in ICRS frame [kpc].
            - 'pm_ra' (np.ndarray): Proper motion in RA [mas/yr].
            - 'pm_dec' (np.ndarray): Proper motion in Dec [mas/yr].
            - 'l_gal' (np.ndarray): Galactic longitude [deg].
            - 'b_gal' (np.ndarray): Galactic latitude [deg].
            - 'pm_l' (np.ndarray): Proper motion in Galactic longitude [mas/yr].
            - 'pm_b' (np.ndarray): Proper motion in Galactic latitude [mas/yr].
    """

    r = dict_pop_dyn["r"]
    phi = dict_pop_dyn["phi"]
    z = dict_pop_dyn["z"]
    v_r = dict_pop_dyn["v_r"]
    v_phi = dict_pop_dyn["v_phi"]
    v_z = dict_pop_dyn["v_z"]

    # Convert from polar coordinates to Cartesian coordinates.
    x, y = polar_to_cartesian(r, phi)

    # Convert velocity components from galactocentric cylindrical coordinates
    # to galactocentric Cartesian coordinates.
    (
        v_x,
        v_y,
        v_z,
    ) = speed_cylindrical_to_cartesian(v_r, v_phi, v_z, phi)

    # Convert galactocentric coordinates and velocities into ICRS and galactic frame.
    (
        ra,
        dec,
        sun_dist_icrs,
        pm_ra,
        pm_dec,
        v_ls_icrs,
    ) = galactocentric_to_icrs(x, y, z, v_x, v_y, v_z)

    (
        l_gal,
        b_gal,
        sun_dist_gal,
        pm_l,
        pm_b,
        v_ls_gal,
    ) = galactocentric_to_galactic(x, y, z, v_x, v_y, v_z)

    dict_pop_dyn["ra"] = ra
    dict_pop_dyn["dec"] = dec
    dict_pop_dyn["dist"] = sun_dist_icrs
    dict_pop_dyn["pm_ra"] = pm_ra
    dict_pop_dyn["pm_dec"] = pm_dec
    dict_pop_dyn["l"] = l_gal
    dict_pop_dyn["b"] = b_gal
    dict_pop_dyn["v_ls"] = v_ls_icrs

    return dict_pop_dyn
