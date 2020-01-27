"""
Initial galactocentric position for the stellar population.

We follow Faucher-Giguère & Kaspi (2006) and choose a galactocentric coordinate system,
where the galactic centre is located at the origin. In terms of galactic latitude l and
longitude b, the x-,y-, and z-axes are parallel to (l, b) = (90, 0), (180, 0) and (0,
90), respectively, forming a right-handed Cartesian frame. Moreover, we define r =
(x**2 + y**2)**0.5 as the distance from the galactic centre in the galactic plane and
theta = arctan(y/x)
"""


from typing import Tuple

import numpy as np

import pypopsyn.simulator.coordinate_conversions as coco


def check_arm_index(arm_index: int):
    """
    Check that the index for the spiral galaxy arms is not <1 or >4.

    Args:
        arm_index (int): index for the respective spiral arms
    """
    if arm_index < 1 or arm_index > 4:
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
    coco.check_radial_coordinate(r)

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


def pdf_initial_coordinates(r: float, arm_index: int) -> Tuple[float, float]:
    """
    Probability density function for stellar galactocentric position incorporating
    the Milky Way's arm structure based on Faucher-Giguère & Kaspi (2006) (see also
    Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc
        arm_index (int): index for the respective spiral arms, 0 < arm_index < 5

    Returns:
        (float, float): galactocentric coordinates theta [rad], r [kpc] with noise
    """

    # check range of input
    coco.check_radial_coordinate(r)
    check_arm_index(arm_index)

    theta = calculate_theta(r, arm_index)
    theta_corr, r_corr = calculate_noise_for_coordinates(r)

    theta = theta + theta_corr
    r = r + r_corr

    return theta, r


def calculate_theta(r: float, arm_index: int) -> float:
    """
    Calculating the angular coordinate of a neutron star for a given distance
    from the galactic centre incorporating the Milky Way's arm structure from
    Faucher-Giguère & Kaspi (2006) (see also Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc
        arm_index (int): index for the respective spiral arms, 0 < arm_index < 5

    Returns:
        float: galactocentric theta coordinate in rad
    """

    # check range of input
    coco.check_radial_coordinate(r)
    check_arm_index(arm_index)

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

    theta = (
        arm_param[arm_index][0] * np.log(r / arm_param[arm_index][1])
        + arm_param[arm_index][2]
    )

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
    Probability density function for the height from the galactic equatorial plane.

    Args:
        z (float): distance from the galactic plane in kpc

    Returns:
        float: distribution of stars per kpc in z direction
    """

    # we use an exponential distribution as given by Wainscoat et al. (1992)
    # and choose a mean scale height characteristic for a young distribution as
    # obtained by Gullon et al. (2014)

    h_mean = 0.1  # [kpc]
    p_z = 1.0 / h_mean * np.exp(-z / h_mean)

    return p_z


def random_scatter_about_plane(
    z: np.ndarray, NS_number: int, seed: int = None
) -> np.ndarray:
    """
    Randomly distribute positive height values within z about the galactic plane
    located at z=0.

    Args:
        z (np.nparray): array of heights in kpc with positive values
        NS_number (int): total number of neutron stars created in the simulation
        seed (int): seed for random number generation,
                    set to None unless otherwise specified

    Returns:
        (np.nparray): array of heights in kpc randomly scattered above or below 0
    """

    np.random.seed(seed)

    # check that z has the length of the number of neutron stars simulated
    if len(z) != NS_number:
        raise ValueError("Input array has the wrong length")

    # for each neutron star create a random value 0 or 1 (above or below plane)

    up_down_index = np.random.randint(0, 2, NS_number)
    z_rand = np.zeros(NS_number)

    for i in range(NS_number):
        if up_down_index[i] == 0:
            z_rand[i] = z[i]
        else:
            z_rand[i] = -z[i]

    return z_rand
