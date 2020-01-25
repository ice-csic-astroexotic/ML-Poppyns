"""Structure and stellar surface density in the Milky Way.

We follow Faucher-Giguère & Kaspi (2006) and choose a galactocentric coordinate system, where the galactic center is located at the origin. In terms of galactic latitude l and longitude b, the x-,y-, and z-axes are parallel to (l, b) = (90, 0),
(180, 0) and (0, 90), respectively, forming a right-handed Cartesian frame.
Moreover, we define r = (x**2 + y**2)**0.5 as the distance from the galactic center
in the galactic plane and theta = arctan(y/x)"""


from typing import Tuple

import numpy as np

# constants
zsun = 0.02  # Sun's distance from the galactic plane [kpc]


def stellar_surf_density(r: float) -> float:
    """
    Milky Way's stellar surface density according to Eqn. (15) of Yusifov & Küçük (2004).

    Args:
        r (float): distance from the galactic centre in kpc

    Returns:
        float: stellar surface density in 1/kpc**2
    """

    if r < 0:
        raise ValueError("radial coordinate is out of range")

    rsun = 8.5  # Sun's distance from the galactic center [kpc]
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


def pdf_milky_way(r: float, i: int) -> Tuple[float, ...]:
    """
    Probability distribution function for stellar galactic coordinates incorporating
    the Milky Way's arm structure based on Faucher-Giguère & Kaspi (2006)
    (see also Wainscoat et al. (1992)).

    Args:
        r (float): distance from the galactic centre in kpc
        i (int): index for the respective spiral arms

    Returns:
        (float, float): galactocentric coordinates theta and r
    """

    if r < 0:
        raise ValueError("radial coordinate is out of range")

    if i < 1 or i > 4:
        raise ValueError("arm index is out of range")

    # parameters of the four spiral arms in the Milky Way according to Table 2 in Faucher-Giguère
    # & Kaspi giving winding constant k [rad], inner radius r_0 [kpc] and inner angle theta_min
    # [rad] for the Norma, Carina-Sagittarius, Perseus, and Crux-Scutum arm

    arm_param = {
        "1": np.array([4.25, 3.48, 1.57]),
        "2": np.array([4.25, 3.48, 4.71]),
        "3": np.array([4.89, 4.90, 4.09]),
        "4": np.array([4.89, 4.90, 0.95]),
    }

    index = str(i)
    theta = (
        arm_param[index][0] * np.log(r / arm_param[index][1])
        + arm_param[index][2]
    )

    # adding noise to the radial and angular coordinate to avoid artificial features near the galactic centre;
    # see Sec. 3.2.1 in Faucher-Giguère & Kaspi (2006) for details

    theta_corr = np.random.uniform(0, 2 * np.pi) * np.exp(-0.35 * r)
    r_corr = np.random.normal(0, 0.07 * r)

    theta = theta + theta_corr
    r = r + r_corr

    return theta, r
