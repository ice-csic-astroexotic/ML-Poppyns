"""Structure and stellar surface density in the Milky Way.

We follow Faucher-Giguère & Kaspi (2006) and choose a galactocentric coordinate system,
where the galactic centre is located at the origin. In terms of galactic latitude l and
longitude b, the x-,y-, and z-axes are parallel to (l, b) = (90, 0), (180, 0) and (0,
90), respectively, forming a right-handed Cartesian frame. Moreover, we define r =
(x**2 + y**2)**0.5 as the distance from the galactic centre in the galactic plane and
theta = arctan(y/x)"""


from typing import Tuple

import numpy as np


def check_radial_coordinate(r: float):
    """
    Check that the distance from the galactic centre is not negative.

    Args:
        r (float): distance from the galactic centre in kpc
    """
    if r < 0:
        raise ValueError("Radial coordinate is out of range")


def check_arm_index(i: int):
    """
    Check that the index for the spiral galaxy arms is not <1 or >4.

    Args:
        i (int): index for the respective spiral arms
    """
    if i < 1 or i > 4:
        raise ValueError("Arm index is out of range")


def stellar_surf_density(r: float) -> float:
    """
    Milky Way's stellar surface density in the galactic plane according to Eqn. (15)
    of Yusifov & Küçük (2004).

    Args:
        r (float): distance from the galactic centre in kpc

    Returns:
        float: stellar surface density in 1/kpc**2
    """

    # check range of input
    check_radial_coordinate(r)

    rsun = 8.5  # Sun's distance from the galactic centre [kpc]
    A = 37.6  # +- 1.90 [1/kpc**2]
    a = 1.64  # +-0.11
    b = 4.01  # +-0.24
    r1 = 0.55  # +- 0.10 [kpc]

    rho = (
        A
        * ((r + r1) / (rsun + r1)) ** a
        * np.exp(-b * (r - rsun) / (rsun + r1))
    )

    return rho


def pdf_initial_coordinates(r: float, i: int) -> Tuple[float, float]:
    """
    Probability distribution function for stellar galactocentric position incorporating
    the Milky Way's arm structure based on Faucher-Giguère & Kaspi (2006) (see also
    Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc
        i (int): index for the respective spiral arms, 0 < i < 5

    Returns:
        (float, float): galactocentric coordinates theta [rad], r [kpc] with noise
    """

    # check range of input
    check_radial_coordinate(r)
    check_arm_index(i)

    theta = calculate_theta(r, i)
    theta_corr, r_corr = calculate_noise_for_coordinates(r)

    theta = theta + theta_corr
    r = r + r_corr

    return theta, r


def calculate_theta(r: float, i: int) -> float:
    """
    Calculating the angular coordinate of a neutron star for a given distance
    from the galactic centre incorporating the Milky Way's arm structure from
    Faucher-Giguère & Kaspi (2006) (see also Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc
        i (int): index for the respective spiral arms, 0 < i < 5

    Returns:
        float: galactocentric theta coordinate in rad
    """

    # check range of input
    check_radial_coordinate(r)
    check_arm_index(i)

    # parameters for four spiral arms in the Milky Way according to Table 2 in
    # Faucher-Giguère & Kaspi giving the winding constant k [rad], inner radius r_0
    # [kpc] and inner angle theta_min [rad] for the Norma, Carina-Sagittarius,
    # Perseus and Crux-Scutum arm

    arm_param = {
        1: np.array([4.25, 3.48, 1.57]),
        2: np.array([4.25, 3.48, 4.71]),
        3: np.array([4.89, 4.90, 4.09]),
        4: np.array([4.89, 4.90, 0.95]),
    }

    theta = arm_param[i][0] * np.log(r / arm_param[i][1]) + arm_param[i][2]

    return theta


def calculate_noise_for_coordinates(
    r: float, seed: int = None
) -> Tuple[float, float]:
    """
    Calculating noise for the angular and radial coordinate to smear out the
    distribution and avoid artificial features near the galactic centre;
    see Sec. 3.2.1 in Faucher-Giguère & Kaspi (2006) for details.

    Args:
        r (float): distance from the galactic centre in kpc
        seed (int): seed for random number generation,
                    set to None unless otherwise specified

    Returns:
        (float, float): noise for galactocentric coordinates theta [rad], r [kpc]
    """

    np.random.seed(seed)

    theta_corr = np.random.uniform(0, 2 * np.pi) * np.exp(-0.35 * r)
    r_corr = np.random.normal(0, 0.07 * r)

    return theta_corr, r_corr


def pdf_initial_height(z: float) -> float:
    """
    Probability distribution function for the height from the galactic equatorial plane.

    Args:
        z (float): distance from the galactic plane in kpc

    Returns:
        float: distribution of stars per kpc in z direction
    """

    # we use an exponential distribution as given by Wainscoat et al. (1992)
    # and choose a scale height characteristic for a young distribution as obtained
    # by Gullon et al. (2006)

    scale_height = 0.1  # [kpc]
    rho_z = 1.0 / scale_height * np.exp(-z / scale_height)

    return rho_z
