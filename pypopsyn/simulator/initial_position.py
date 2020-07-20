"""
Initial galactocentric position for the stellar population

We follow Faucher-Giguère & Kaspi (2006) and choose a galactocentric coordinate system,
where the galactic centre is located at the origin. In terms of galactic latitude l and
longitude b, the x-,y-, and z-axes are parallel to (l, b) = (90, 0), (180, 0) and (0,
90), respectively, forming a right-handed Cartesian frame. This implies that the Sun is
positioned at (x=0, y=8.5 kpc).

Moreover, we define r = (x**2 + y**2)**0.5 as the distance from the galactic centre in
the galactic plane and phi = arctan(y/x). Here, the angle phi is the same as theta in
Faucher-Giguère & Kaspi (2006). We reserve the variable theta for the polar angle in a
spherical coordinate system.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""


from typing import Tuple

import numpy as np

import pypopsyn.simulator.coordinate_conversions as coco
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
        raise ValueError("Arm index is out of range")


def pdf_radial_stellar_density(r: float) -> float:
    """
    The Milky Way's stellar radial density in the galactic plane according
    to eq. (15) of Yusifov & Küçük (2004).

    Args:
        r (float): distance from the galactic centre in kpc.

    Returns:
        float: stellar radial density in 1/kpc.
    """

    # check range of input
    coco.check_radial_coordinate(r)

    # Here we keep R_sun = 8.5 kpc for consistency with the results of Yusifov & Küçük (2004)
    rsun = 8.5  # Sun's distance from the galactic centre [kpc].
    A = 37.6  # +- 1.90 [1/kpc**2]
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


def pdf_initial_coordinates(r: float, arm_index: int) -> Tuple[float, float]:
    """
    Probability density function for stellar galactocentric position incorporating
    the Milky Way's arm structure based on Faucher-Giguère & Kaspi (2006) (see also
    Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc.
        arm_index (int): index for the respective spiral arms, 0 < arm_index < 5.

    Returns:
        (float, float): galactocentric coordinates phi [rad], r [kpc] with noise.
    """

    # Check range of input.
    coco.check_radial_coordinate(r)
    check_arm_index(arm_index)

    phi = calculate_phi(r, arm_index)
    phi_corr, r_corr = calculate_noise_for_coordinates(r)

    phi = phi + phi_corr
    r = r + r_corr

    return phi, r


def calculate_phi(r: float, arm_index: int) -> float:
    """
    Calculating the angular coordinate of a neutron star for a given distance
    from the galactic centre incorporating the Milky Way's arm structure according
    to eq. (12) of Faucher-Giguère & Kaspi (2006) (see also Wainscoat et al. (1992)).
    Two different parameter sets for the spiral arms are provided. The first one is taken
    from Faucher-Giguère & Kaspi (2006) and the second one from Yao, Manchester & Wang (2017).

    Args:
        r (float): distance from the galactic centre in kpc.
        arm_index (int): index for the respective spiral arms, 0 < arm_index < 5.

    Returns:
        float: galactocentric phi coordinate in rad.
    """

    # Check range of input.
    coco.check_radial_coordinate(r)
    check_arm_index(arm_index)

    # Parameters for four spiral arms in the Milky Way giving the winding constant k [rad], inner radius r_0
    # [kpc] and inner angle phi_min [rad] for the Norma, Carina-Sagittarius, Perseus and Crux-Scutum arm.
    # Parameters are given for two models 1) saFK06 according to Table 2 in Faucher-Giguère & Kaspi (2006) and 2) saYMW17
    # according to table 1 in Yao, Manchester & Wang (2017). In this last case the parameters are adapted to match the same
    # functional form as in Faucher-Giguère & Kaspi (2006).

    arms_pattern = cfg["spiral_arms"]
    if arms_pattern == "saFK06":
        arm_param = {
            1: np.array([4.25, 3.48, 1.57]),
            2: np.array([4.25, 3.48, 4.71]),
            3: np.array([4.89, 4.90, 4.09]),
            4: np.array([4.89, 4.90, 0.95]),
        }
    elif arms_pattern == "saYMW17":
        arm_param = {
            1: np.array([4.95, 3.35, 0.77]),
            2: np.array([5.46, 3.56, 3.82]),
            3: np.array([5.77, 3.71, 2.09]),
            4: np.array([5.37, 3.67, 5.76]),
        }
    else:
        raise ValueError(
            "The spiral pattern does not exist. Choose between saFK06 or saYMW17."
        )

    phi = (
        arm_param[arm_index][0] * np.log(r / arm_param[arm_index][1])
        + arm_param[arm_index][2]
    )

    return phi


def spiral_arm_time_evol(phi0: float, t: float) -> float:
    """
    Evolving the spiral arm position backward for a time t. We assume that the
    galactic spiral structure rotates rigidly in clockwise direction with a
    period of 250 Myr (see 'A guided map to the spiral arms in the galactic disk
    of the Milky Way' by Vallée (2017)).

    Args:
        phi0 (float): current angular position in rad for the current spiral pattern.
        t (float): time in yr to propagate backward.

    Returns:
        (float): angular position in rad for the spiral pattern as it was t years ago.
    """

    # Evaluate the angular velocity of rotation of the spiral pattern;
    # T is the period of rotation in years.
    T = 2.5e8
    omega_spiral_arms = 2.0 * np.pi / T

    # Find the value of the angle phi t years ago.
    phi_t = phi0 + omega_spiral_arms * t

    return phi_t


def calculate_noise_for_coordinates(
    r: float, seed: int = None
) -> Tuple[float, float]:
    """
    Calculating noise for the angular and radial coordinate to smear out the
    distribution and avoid artificial features near the galactic centre;
    see Sec. 3.2.1 in Faucher-Giguère & Kaspi (2006) for details.

    Args:
        r (float): distance from the galactic centre in kpc.
        seed (int): seed for random number generation,
        set to None unless otherwise specified.

    Returns:
        (float, float): noise for galactocentric coordinates phi [rad], r [kpc].
    """

    np.random.seed(seed)

    phi_corr = np.random.uniform(0, 2 * np.pi) * np.exp(-0.35 * r)
    r_corr = np.random.normal(0, 0.07 * r)

    return phi_corr, r_corr


def pdf_initial_height(z: float) -> float:
    """
    Probability density function for the height from the galactic equatorial plane
    according to eq. (2) in Gullon et al. (2014).

    Args:
        z (float): distance from the galactic plane in kpc.

    Returns:
        float: distribution of stars per kpc in z direction.
    """

    # We use an exponential distribution as given by Wainscoat et al. (1992)
    # and choose a mean scale height characteristic for a young distribution as
    # obtained by Gullon et al. (2014).

    h_mean = cfg["h_mean"]
    pdf_z = 1.0 / h_mean * np.exp(-z / h_mean)

    return pdf_z


def random_scatter_about_plane(
    z: np.ndarray, NS_number: int, seed: int = None
) -> np.ndarray:
    """
    Randomly distribute positive height values within z about the galactic plane
    located at z=0.

    Args:
        z (np.ndarray): array of heights in kpc with positive values.
        NS_number (int): total number of neutron stars created in the simulation.
        seed (int): seed for random number generation,
        set to None unless otherwise specified.

    Returns:
        (np.ndarray): array of heights in kpc randomly scattered above or below 0.
    """

    np.random.seed(seed)

    # Check that z has the length of the number of neutron stars simulated.
    if len(z) != NS_number:
        raise ValueError("Input array has the wrong length")

    # For each neutron star create a random value 0 or 1 (above or below plane).

    up_down_index = np.random.randint(0, 2, NS_number)
    z_rand = np.zeros(NS_number)

    for i in range(NS_number):
        if up_down_index[i] == 0:
            z_rand[i] = z[i]
        else:
            z_rand[i] = -z[i]

    return z_rand
