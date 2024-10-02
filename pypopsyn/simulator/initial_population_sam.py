"""
    Generating an initial population of neutron stars in the Milky Way with
    random parameters. For the initial positions we assume that the distribution
    of progenitors follows a given radial distribution and the spiral arms with a given
    parametrized shape.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import logging
from typing import Tuple

import numpy as np

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.magneto_rotational_physics.initial_period as ipd
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.initial_position as ip
import pypopsyn.simulator.stellar_dynamics.initial_velocity as iv
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
import utilities.benchmark.pyinstrument as benchmark
import utilities.samplers.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way.
    """

    def __init__(self, NS_number: int) -> None:
        """
        Initialization for the initial population synthesis.

        Args:
            NS_number (int): Number of neutron stars to simulate.
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
            (np.ndarray): Array of ages in [yr].
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
        self, t_age: np.ndarray, spiral_model: sm.SpiralModelBase
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in
        cylindrical reference frame.

        Args:
            t_age (np.ndarray): Array of neutron star ages in [yr].
            spiral_model (sm.SpiralModelBase): A class specifying the spiral arm structure model.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Polar r, phi and z coordinates in [kpc], [rad] and [kpc]
                respectively for each generated neutron star.
        """

        radial_model = cfg["radial_model"]
        if radial_model == "rmYK04":
            pdf_radial = ip.pdf_radial_density_YK04
        elif radial_model == "rmVV21":
            pdf_radial = ip.pdf_radial_density_VV21
        else:
            raise ValueError(
                "The radial density model pdf does not exist. Choose between rmYK04 or rmVV21."
            )

        # Randomly associate a spiral arm to each neutron star.
        arm_index_rand = spiral_model.generate_arm_index(
            cfg["arm_number"], self.NS_number
        )
        # Count the number of stars in the Local arm.
        NS_local = len(arm_index_rand[arm_index_rand == 5])

        # Drawing a random distance from the galactic center in [kpc] for
        # each neutron star according to the radial stellar density.
        r_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["r_extent"]), cfg["resolution"]
        )

        r_pdf_rand = np.zeros(self.NS_number)
        r_pdf_rand[arm_index_rand != 5] = rs.random_from_pdf(
            r_grid, pdf_radial, self.NS_number - NS_local
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
        phi_rand, r_rand = ip.smear_initial_coordinates(
            r_pdf_rand, phi, self.NS_number
        )

        # Propagating the azimuthal coordinate of each object backwards in time
        # (according to its age) to account for the rotation of the galactic arms;
        # we assume that the arm structure itself remains rigid.
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
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): vk_r, vk_phi and vk_z kick
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
            r (np.ndarray): Distance in the galactic disk from the galactic center
                in [kpc].
            z (np.ndarray): Height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): Array of orbital velocities in [kpc/yr].
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
            (np.ndarray): Initial spin periods of the pulsar sample in [s].
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
            (np.ndarray): Initial magnetic field strengths of the pulsar sample in [G].
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
            (np.ndarray): Initial misalignment angles of the pulsar sample in [rad].
        """

        chi_grid = np.linspace(0.0, np.pi / 2, cfg["resolution"])
        chi_rand = rs.random_from_pdf(chi_grid, np.sin, self.NS_number)

        return chi_rand
