"""
Generating an initial population of neutron stars in the Milky Way with
random parameters. For the initial positions we assume that the distribution
of progenitors follows the free electron density model ymw16 from Yau et al. (2016).

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

import logging
import pathlib
from typing import Tuple

import numpy as np

import pypopsyn.benchmark.pyinstrument as benchmark
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.magneto_rotational_physics.initial_period as ipd
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.initial_position as ip
import pypopsyn.simulator.stellar_dynamics.initial_velocity as iv
import utilities.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way.
    """

    def __init__(self, NS_number: int = cfg["NS_number"]) -> None:
        """
        Initialization for the initial population synthesis.
        """

        # Number of neutron stars to generate in a single call of the InitialNeutronStarPopulation class.
        # The default value corresponds to the total number specified in the configuration file, so that
        # all neutron stars of the population are generated at once.
        # For a one by one simulation NS_number is set to 1 and the simulator calls this class repeatedly
        # generating stars in a loop until the wanted total number of objects in the population is reached.
        self.NS_number = NS_number

    def age(self) -> np.ndarray:
        """
        Drawing a random age in [yr] for each neutron star from a uniform
        probability distribution in a given range of time.

        Returns:

            np.ndarray: array of ages in [yr].

        """

        log.debug(
            "Drawing random age in range [{},{}]".format(
                cfg["t_age_min"], cfg["t_age_max"]
            )
        )

        t_age = np.random.uniform(
            cfg["t_age_min"], cfg["t_age_max"], self.NS_number
        )
        return t_age

    @benchmark.profile(
        enabled=cfg["enable_profiles"],
        show=cfg["show_profiles"],
        output_dir=cfg["profiles_dir"],
    )
    def position(
        self, t_age: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in
        cylindrical reference frame according to the Galactic electron density
        distribution ymw16 (see Yau et al. 2016).
        Using the notebook ns_distribution_ne_model.ipynb we create a 2D numpy array containing the
        electron density distribution in polar coordinates (r, phi).
        This 2D array is used to sample the neutron star positions in the Galaxy.

        Args:

            t_age (np.ndarray): array of neutron star ages in [yr].

        Returns:

            (np.ndarray, np.ndarray, np.ndarray):
            polar r, phi and z coordinates in [kpc], [rad] and [kpc] respectively
            for each generated neutron star.
        """

        # Load the neutron star density model table.
        # The model table has been generated through the Jupyter notebook ns_distributio_ne_model.ipynb.
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
            r_grid, phi_grid, NS_density_model, self.NS_number
        )

        # Propagating the azimuthal coordinate of each object backwards in time
        # (according to its age) to account for the rotation of the galactic arms;
        # we assume that the density structure itself remains rigid.
        phi_rand = ip.spiral_arm_time_evol(phi_rand, t_age)

        # Drawing a random distance from the galactic plane in [kpc] for each neutron
        # star according to the probability density function for the height.
        z_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["z_extent"]), cfg["resolution"]
        )
        z_pdf_rand = rs.random_from_pdf(
            z_grid, ip.pdf_initial_height, self.NS_number
        )

        # Randomly distribute the stars above and below the galactic plane.
        z_rand = ip.random_scatter_about_plane(z_pdf_rand, self.NS_number)

        return r_rand, phi_rand, z_rand

    def kick_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the kick velocity of each random neutron star in a cylindrical
        galactocentric coordinate system.

        Returns:

            (np.ndarray, np.ndarray, np.ndarray): vk_r, vk_phi and vk_z kick
            velocities in [kpc/yr] for each generated neutron stars. In particular
            vk_r is the component of the kick velocity along the galactocentric
            radial direction, vk_phi is the component along the azimuthal phi
            direction and vk_z is the component along the z direction.

        """

        kick_model = cfg["kick_model"]
        if kick_model == "km_maxwell":
            pdf_vkick = iv.pdf_kick_velocity_maxwell
        elif kick_model == "km_exp":
            pdf_vkick = iv.pdf_kick_velocity_exp
        elif kick_model == "km_2maxwell":
            pdf_vkick = iv.pdf_kick_velocity_2maxwell
        else:
            raise ValueError(
                "The kick velocity model pdf does not exist. Choose between km_maxwell, km_exp or km_2maxwell."
            )

        # Drawing a random magnitude of the birth kick velocity in [km/s] for each
        # neutron star according to the underlying velocity probability density
        # function.
        vk_grid = np.linspace(0.0, cfg["vk_extent"], cfg["resolution"])
        vk_rand = rs.random_from_pdf(vk_grid, pdf_vkick, self.NS_number)
        # Convert from [km/s] to [kpc/yr].
        vk_rand = vk_rand * const.YR_TO_S / const.KPC_TO_KM

        # To draw a random direction for the speed from a uniform distribution,
        # we uniformly sample the azimuthal angle [rad] in the range [0, 2*np.pi];
        # to obtain a uniform distribution in the z-direction, we sample the polar
        # angle [rad] in the range [0, np.pi] according to the PDF np.sin.
        psi_rand = np.random.uniform(0, 2 * np.pi, self.NS_number)
        theta_grid = np.linspace(0.0, np.pi, cfg["resolution"])
        theta_rand = rs.random_from_pdf(theta_grid, np.sin, self.NS_number)

        # Project the velocity on a Cartesian reference frame co-moving with each
        # star, where the local x-axis points always in the r-direction of our
        # galactocentric frame, the local y-axis in the azimuthal phi-direction
        # and the local z-axis coincides with the galactocentric one.
        vk_r_rand, vk_phi_rand, vk_z_rand = coco.spherical_to_cartesian(
            vk_rand, theta_rand, psi_rand
        )

        return vk_r_rand, vk_phi_rand, vk_z_rand

    @staticmethod
    def orbital_velocity(r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate the orbital circular velocity of each star in the galactic
        gravitational potential. In galactocentric cylindrical coordinates,
        the only non-zero component is the azimuthal phi component. Since the stars
        in the galaxy rotate in the clockwise direction, i.e., towards decreasing phi
        values, the phi component is negative.

        Args:

            r (np.ndarray): distance in the galactic disk from the galactic center
            in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:

            (np.ndarray): array of orbital velocities in [kpc/yr].

        """
        circular_velocity_vect = np.vectorize(iv.circular_velocity)
        v_orb = -circular_velocity_vect(r, z)

        return v_orb

    def period(self) -> np.ndarray:
        """
        Determining the initial rotation periods of each pulsar in the sample,
        as drawn from a log-normal distribution. The characteristic
        parameters are defined in config_simulator.py.

        Returns:

            (np.ndarray): initial spin periods of the pulsar sample in [s].

        """

        spin_period_model = cfg["spin_period_model"]

        if spin_period_model == "normal":
            P_rand = ipd.pdf_period_normal(
                cfg["P_initial_mean"],
                cfg["P_initial_sigma"],
                self.NS_number,
            )
        elif spin_period_model == "log-normal":
            P_rand = ipd.pdf_period_lognormal(
                cfg["P_initial_log10_mean"],
                cfg["P_initial_log10_sigma"],
                self.NS_number,
            )
        else:
            raise ValueError(
                "The initial spin-period model pdf does not exist. Choose between normal or log-normal."
            )

        return P_rand

    def magnetic_field(self) -> np.ndarray:
        """
        We follow Faucher-Giguère & Kaspi (2006) and Gullon et al. (2014) and determine the
        initial magnetic field strengths of each pulsar in the sample, by drawing values from
        a log-normal distribution, i.e., the log_10 values of the magnetic field strengths are
        themselves normally distributed. The characteristic parameters are defined in config_simulator.py.

        Returns:

            (np.ndarray): initial magnetic field strengths of the pulsar sample in [G].

        """

        B_rand = 10 ** np.random.normal(
            cfg["B_initial_log10_mean"],
            cfg["B_initial_log10_sigma"],
            self.NS_number,
        )

        return B_rand

    def misalignment_angle(self) -> np.ndarray:
        """
        We follow Gullon et al. (2014) and choose the initial misalignment angle in the
        range [0, np.pi / 2] according to the probability density distribution np.sin.

        Returns:

            (np.ndarray): initial misalignment angles of the pulsar sample in [rad].

        """

        chi_grid = np.linspace(0.0, np.pi / 2, cfg["resolution"])
        chi_rand = rs.random_from_pdf(chi_grid, np.sin, self.NS_number)

        return chi_rand
