"""
Initial galactocentric position for the stellar population.

We follow Faucher-Giguère & Kaspi (2006) and choose a galactocentric coordinate system,
where the galactic center is located at the origin. In terms of galactic latitude l and
longitude b, the x-,y-, and z-axes are parallel to (l, b) = (90, 0), (180, 0) and (0,
90), respectively, forming a right-handed Cartesian frame. This implies that the Sun is
positioned at (x=0, y=8.5 kpc).

Moreover, we define r = (x^2 + y^2)^0.5 as the distance from the galactic center in
the galactic plane and phi = arctan(y/x). Here, the angle phi is the same as theta in
Faucher-Giguère & Kaspi (2006). We reserve the variable theta for the polar angle in a
spherical coordinate system.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

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


from typing import Tuple

import numpy as np

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
from pypopsyn.simulator.configuration import cfg


def check_arm_index(arm_index: int) -> None:
    """
    Check that the index for the spiral galaxy arms is not <1 or >4.

    Args:
        arm_index (int): index for the respective spiral arms.

    Returns:
        Returns None if arm_index between or equal to 1 and 4,
        otherwise raises ValueError.
    """
    if arm_index < 1 or arm_index > 4:
        raise ValueError("Arm index is out of range.")


def pdf_radial_stellar_density(r: float) -> float:
    """
    The Milky Way's stellar radial density in the galactic plane according
    to eq. (15) of Yusifov & Küçük (2004).

    Args:
        r (float): distance from the galactic center in [kpc].

    Returns:
        float: stellar radial density in [1/kpc].
    """

    # check range of input
    coco.check_radial_coordinate(r)

    # Here we keep R_sun = 8.5 kpc for consistency with the results
    # of Yusifov & Küçük (2004)
    rsun = 8.5  # Sun's distance from the galactic center in [kpc].
    A = 37.6  # +- 1.90 [1/kpc^2]
    a = 1.64  # +-0.11
    b = 4.01  # +-0.24
    r1 = 0.55  # +- 0.10 [kpc]

    # Stellar surface density following eq. (15) of Yusifov & Küçük (2004).
    rho = (
        A
        * ((r + r1) / (rsun + r1)) ** a
        * np.exp(-b * (r - rsun) / (rsun + r1))
    )

    # Multiply the stellar surface density with the area element in polar coordinates.
    pdf_r = 2 * np.pi * r * rho

    return pdf_r


def pdf_initial_coordinates(
    r: np.ndarray, NS_number: int, arm_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability density function for stellar galactocentric position incorporating
    the Milky Way's arm structure based on Faucher-Giguère & Kaspi (2006) (see also
    Wainscoat et al. 1992).

    Args:
        r (np.ndarray): distances from the galactic center in [kpc].
        NS_number (int): total number of neutron stars created in the simulation.
        arm_index (np.ndarray): indices for the respective spiral arms,
        0 < arm_index < 5.

    Returns:
        (np.ndarray, np.ndarray): galactocentric coordinates phi [rad], r [kpc] with
        noise applied.
    """

    # Check range of input.
    check_radial_coordinate_vect = np.vectorize(coco.check_radial_coordinate)
    check_radial_coordinate_vect(r)
    check_arm_index_vect = np.vectorize(check_arm_index)
    check_arm_index_vect(arm_index)

    phi = calculate_phi(r, arm_index)
    phi_corr, r_corr = calculate_noise_for_coordinates(r, NS_number)

    phi = phi + phi_corr
    r = r + r_corr

    return phi, r


def calculate_phi(r: np.ndarray, arm_index: np.ndarray) -> np.ndarray:
    """
    Calculating the angular coordinates of a neutron star for a given distance
    from the galactic center incorporating the Milky Way's arm structure according
    to eq. (12) of Faucher-Giguère & Kaspi (2006) (see also Wainscoat et al. 1992).
    Two different parameter sets for the spiral arms are provided. The first one is
    taken from Faucher-Giguère & Kaspi (2006) and the second one from Yao, Manchester
    & Wang (2017).

    Args:
        r (np.ndarray): distances from the galactic center in [kpc].
        arm_index (np.ndarray): indices for the respective spiral arms,
        0 < arm_index < 5.

    Returns:
        np.ndarray: galactocentric phi coordinates in [rad].
    """

    # Check range of input.
    check_radial_coordinate_vect = np.vectorize(coco.check_radial_coordinate)
    check_radial_coordinate_vect(r)
    check_arm_index_vect = np.vectorize(check_arm_index)
    check_arm_index_vect(arm_index)

    # Parameters for four spiral arms in the Milky Way giving the winding constant k
    # [rad], inner radius r_0 [kpc] and inner angle phi_min [rad] for the Norma,
    # Carina-Sagittarius, Perseus and Crux-Scutum arm. Parameters are given for two
    # models 1) saFK06 according to Table 2 in Faucher-Giguère & Kaspi (2006) and
    # 2) saYMW17 according to Table 1 in Yao, Manchester & Wang (2017). In this last
    # case the parameters are adapted to match the same functional form as in
    # Faucher-Giguère & Kaspi (2006).

    arms_pattern = cfg["spiral_arms"]

    if arms_pattern == "saFK06":
        arm_param = {
            1: (4.25, 3.48, 1.57),
            2: (4.25, 3.48, 4.71),
            3: (4.89, 4.90, 4.09),
            4: (4.89, 4.90, 0.95),
        }
    elif arms_pattern == "saYMW17":
        arm_param = {
            1: (4.95, 3.35, 0.77),
            2: (5.46, 3.56, 3.82),
            3: (5.77, 3.71, 2.09),
            4: (5.37, 3.67, 5.76),
        }
    else:
        raise ValueError(
            "The spiral pattern does not exist. Choose between saFK06 or saYMW17."
        )

    arm_param_vect = np.vectorize(arm_param.get)(arm_index)

    phi = arm_param_vect[0] * np.log(r / arm_param_vect[1]) + arm_param_vect[2]

    return phi


def spiral_arm_time_evol(phi0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evolving the spiral arm positions backward for a given age t. We assume that the
    galactic spiral structure rotates rigidly in clockwise direction with a
    period of 250 Myr (see 'A guided map to the spiral arms in the galactic disk
    of the Milky Way' by Vallée (2017)).

    Args:
        phi0 (np.ndarray): current angular positions in [rad] for the chosen
        spiral pattern.
        t (np.ndarray): times in [yr] to propagate backward.

    Returns:
        (np.ndarray): angular positions in [rad] for the spiral pattern as they were
        t years ago.
    """

    # Evaluate the angular velocity of rotation of the spiral pattern;
    # T is the period of rotation in [yr].
    T = 2.5e8
    omega_spiral_arms = 2.0 * np.pi / T

    # Find the values of the angles phi t years ago.
    phi_t = phi0 + omega_spiral_arms * t

    return phi_t


def calculate_noise_for_coordinates(
    r: np.ndarray, NS_number: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating noise for the angular and radial coordinate to smear out the
    distribution and avoid artificial features near the galactic center;
    see Sec. 3.2.1 in Faucher-Giguère & Kaspi (2006) for details.

    Args:
        r (np.ndarray): array of distances from the galactic center in [kpc].
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray, np.ndarray): array of noise for the galactocentric coordinates
        phi [rad], r [kpc].
    """

    phi_corr = np.random.uniform(0, 2 * np.pi, NS_number) * np.exp(-0.35 * r)
    r_corr = np.random.normal(0, 0.07 * r, NS_number)

    return phi_corr, r_corr


def pdf_initial_height(z: float) -> float:
    """
    Probability density function for the height from the galactic equatorial plane
    according to eq. (2) in Gullon et al. (2014).

    Args:
        z (float): distance from the galactic plane in [kpc].

    Returns:
        float: distribution of stars per kpc in z direction.
    """

    # We use an exponential distribution as given by Wainscoat et al. (1992)
    # and choose a mean scale height characteristic for a young distribution as
    # obtained by Gullon et al. (2014).

    h_c = cfg["h_c"]
    pdf_z = 1.0 / h_c * np.exp(-z / h_c)

    return pdf_z


def random_scatter_about_plane(z: np.ndarray, NS_number: int) -> np.ndarray:
    """
    Randomly distribute positive height values within z about the galactic plane
    located at z=0.

    Args:
        z (np.ndarray): array of heights in [kpc] with positive values.
        NS_number (int): total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): array of heights in [kpc] randomly scattered above or below 0.
    """

    # Check that z has the length of the number of neutron stars simulated.
    if len(z) != NS_number:
        raise ValueError("Input array has the wrong length")

    # For each neutron star determine if it is above (False) or below (True) the galactic plane.
    if_below = np.random.choice([False, True], size=NS_number)
    z[if_below] = -z[if_below]

    return z
