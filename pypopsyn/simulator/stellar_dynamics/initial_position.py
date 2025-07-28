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
"""


import pathlib
from typing import Tuple

import numpy as np

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
import utilities.samplers.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg


def pdf_radial_density_YK04(r: np.ndarray) -> np.ndarray:
    """
    The Milky Way's stellar radial density in the galactic plane according
    to eq. (15) of Yusifov & Küçük (2004).

    Args:
        r (np.ndarray): Distance from the galactic center in [kpc].

    Returns:
        (np.ndarray): Stellar radial density in [1/kpc].

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


def pdf_radial_density_VV21(r: np.ndarray) -> np.ndarray:
    """
    The Milky Way's radial density of supernova remnants including both core-collapse
    and thermonuclear supernovae. The model distribution is the exponential model from eq. (9)
    from the work of Verberne & Vink (2021).

    Args:
        r (np.ndarray): Distance from the galactic center in [kpc].

    Returns:
        (np.ndarray): SNR radial density in [1/kpc].

    """

    # Check range of input.
    coco.check_radial_coordinate(r)

    # Here we keep R_sun = 8.0 kpc for consistency with the results
    # of Verberne & Vink (2021).
    rsun = 8.0  # Sun's distance from the galactic center in [kpc].
    b = 2.46  # +0.39 -0.33

    # SNR surface density following eq. (9) of Verberne & Vink (2021).
    rho = np.exp(-b * (r - rsun) / rsun)

    # Multiply the stellar surface density with the area element in polar coordinates.
    pdf_r = 2 * np.pi * r * rho

    return pdf_r


def smear_initial_coordinates(
    r: np.ndarray, phi: np.ndarray, NS_number: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smear the initial radial and angular coordinates in the galactocentric frame by adding noise.

    Args:
        r (np.ndarray): Distances from the galactic center in [kpc].
        phi (np.ndarray): Azimuthal coordinate of the stars on the spiral arms [rad].
        NS_number (int): Total number of neutron stars created in the simulation.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Galactocentric coordinates phi [rad], r [kpc] with
            noise applied.

    """

    phi_corr, r_corr = calculate_noise_for_coordinates(r, NS_number)

    phi = phi + phi_corr
    r = r + r_corr

    return phi, r


def spiral_arm_time_evol(phi0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evolving the spiral arm positions backward for a given age t. We assume that the
    galactic spiral structure rotates rigidly in clockwise direction with a
    period of 250 Myr (see 'A guided map to the spiral arms in the galactic disk
    of the Milky Way' by Vallée (2017)).

    Args:
        phi0 (np.ndarray): Current angular positions in [rad] for the chosen
            spiral pattern.
        t (np.ndarray): Times in [yr] to propagate backward.

    Returns:
        (np.ndarray): Angular positions in [rad] for the spiral pattern as they were
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
        r (np.ndarray): Array of distances from the galactic center in [kpc].
        NS_number (int): Total number of neutron stars created in the simulation.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Array of noise for the galactocentric coordinates
            phi [rad], r [kpc].

    """

    phi_corr = np.random.uniform(0, 2 * np.pi, NS_number) * np.exp(-0.35 * r)
    r_corr = np.random.normal(0, 0.07 * r, NS_number)

    return phi_corr, r_corr


def pdf_initial_height(z: np.ndarray) -> np.ndarray:
    """
    Probability density function for the height from the galactic equatorial plane
    according to eq. (2) in Gullon et al. (2014).

    Args:
        z (np.ndarray): Distance from the galactic plane in [kpc].

    Returns:
        (np.ndarray): Distribution of stars per kpc in z direction.

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
        z (np.ndarray): Array of heights in [kpc] with positive values.
        NS_number (int): Total number of neutron stars created in the simulation.

    Returns:
        (np.ndarray): Array of heights in [kpc] randomly scattered above or below 0.

    """

    # Check that z has the length of the number of neutron stars simulated.
    if len(z) != NS_number:
        raise ValueError("Input array has the wrong length")

    # For each neutron star determine if it is above (False) or below (True) the galactic plane.
    if_below = np.random.choice([False, True], size=NS_number)
    z[if_below] = -z[if_below]

    return z


def calculate_r_phi_spiral_model(
    t_age: np.ndarray, spiral_model: sm.SpiralModelBase
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating the position at birth of each random neutron star in a cylindrical reference frame for a spiral model.

    Args:
        t_age (np.ndarray): Array of neutron star ages in [yr].
        spiral_model (sm.SpiralModelBase): A class specifying the spiral arm structure model.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Polar r and phi coordinates in [kpc] and [rad], respectively, for each
            generated neutron star.
    """

    radial_model = cfg["radial_model"]
    if radial_model == "rmYK04":
        pdf_radial = pdf_radial_density_YK04
    elif radial_model == "rmVV21":
        pdf_radial = pdf_radial_density_VV21
    else:
        raise ValueError(
            "The radial density model pdf does not exist. Choose between rmYK04 or rmVV21."
        )

    # Randomly associate a spiral arm to each neutron star.
    arm_index_rand = spiral_model.generate_arm_index(
        cfg["arm_number"], len(t_age)
    )
    # Count the number of stars in the Local arm.
    NS_local = len(arm_index_rand[arm_index_rand == 5])

    # Drawing a random distance from the galactic center in [kpc] for
    # each neutron star according to the radial stellar density.
    r_grid = np.logspace(
        np.log10(0.0001), np.log10(cfg["r_extent"]), cfg["resolution"]
    )

    r_pdf_rand = np.zeros(len(t_age))
    r_pdf_rand[arm_index_rand != 5] = rs.random_from_pdf(
        r_grid, pdf_radial, len(t_age) - NS_local
    )

    if NS_local != 0:
        r_grid_local = np.logspace(
            np.log10(spiral_model.local_r_min),
            np.log10(spiral_model.local_r_max),
            cfg["resolution"],
        )

        r_pdf_rand[arm_index_rand == 5] = rs.random_from_pdf(
            r_grid_local, pdf_radial, NS_local
        )

    # Evaluate the angular phi coordinate for each neutron star and
    # add noise to both galactocentric coordinates.
    phi = sm.spiral_model.calculate_phi(r_pdf_rand, arm_index_rand)
    phi_rand, r_rand = smear_initial_coordinates(r_pdf_rand, phi, len(t_age))

    # Propagating the azimuthal coordinate of each object backwards in time
    # (according to its age) to account for the rotation of the galactic arms;
    # we assume that the arm structure itself remains rigid.
    phi_rand = spiral_arm_time_evol(phi_rand, t_age)

    return r_rand, phi_rand


def calculate_r_phi_electron_density(
    t_age: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating the position at birth of each random neutron star in a cylindrical reference frame according to the
    Galactic electron density distribution ymw16 (see Yao et al. 2017).

    Using the notebook ns_distribution_ne_model.ipynb, we create a 2D NumPy array containing the electron density
    distribution in polar coordinates (r, phi). This 2D array is used to sample the neutron star positions in the Galaxy.

    Args:
        t_age (np.ndarray): Array of neutron star ages in [yr].

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Polar r and phi coordinates in [kpc] and [rad], respectively, for each
            generated neutron star.
    """

    # Load the neutron star density model table.
    # The model table has been generated through the Jupyter notebook located in
    # tutorials/analysis_notebooks/ns_distribution_ne_model.ipynb.
    # It contains an 2D array of density rho in cylindrical coordinates (r, phi).
    # The density in the table is already multiplied by the galactocentric distance r
    # to take into account the element of area correction.

    file = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/stellar_dynamics/YMW16_density_model.npy",
    )
    NS_density_model = np.load(file)

    # Define the grid of coordinates.
    r_grid = np.linspace(0.0, cfg["r_extent"], NS_density_model.shape[0])
    phi_grid = np.linspace(0.0, 2.0 * np.pi, NS_density_model.shape[1])

    # Drawing a random distance from the galactic center in [kpc] and a random
    # azimuthal angle in [rad] according to the 2d density model.
    r_rand, phi_rand = rs.random_from_pdf_2d(
        r_grid, phi_grid, NS_density_model, len(t_age)
    )

    # Propagating the azimuthal coordinate of each object backwards in time
    # (according to its age) to account for the rotation of the galactic arms;
    # we assume that the density structure itself remains rigid.
    phi_rand = spiral_arm_time_evol(phi_rand, t_age)

    return r_rand, phi_rand
