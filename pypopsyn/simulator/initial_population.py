"""
    Generating an initial population of neutron stars in the Milky Way with
    random parameters. For the initial positions, we assume that the distribution
    of progenitors follows the free electron density model ymw16 from Yao et al. (2017).

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import logging
from typing import Tuple

import numpy as np

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.magneto_rotational_physics.initial_magnetic_field as imf
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
        self, t_age: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in a cylindrical reference frame according
        either to the Galactic electron density distribution ymw16 (see Yao et al. 2017) when cfg["sample_edm"] = True
        in the `config_simulator.py` file, or using a spiral model and a radial model as specified in the configuration
        file when cfg["sample_edm"] = False.

        Args:
            t_age (np.ndarray): Array of neutron star ages in [yr].

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Polar r, phi and z coordinates in [kpc], [rad] and [kpc]
                respectively for each generated neutron star.
        """

        if cfg["sample_edm"]:
            r_rand, phi_rand = ip.calculate_r_phi_electron_density(t_age)
        else:
            r_rand, phi_rand = ip.calculate_r_phi_spiral_model(
                t_age, sm.spiral_model
            )

        z_rand = ip.calculate_z(self.NS_number)

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

        # Drawing a random magnitude of the birth kick velocity in [km/s] for each
        # neutron star according to the underlying velocity probability density
        # function.
        if kick_model == "km_maxwell":
            vk_rand = iv.kick_velocity_maxwell(self.NS_number)
        elif kick_model == "km_exp":
            vk_rand = iv.kick_velocity_exp(self.NS_number)
        elif kick_model == "km_double_maxwell":
            vk_rand = iv.kick_velocity_double_maxwell(self.NS_number)
        elif kick_model == "km_log-normal":
            vk_rand = iv.kick_velocity_lognormal(
                cfg["vk_ln_mean"], cfg["vk_ln_sigma"], self.NS_number
            )
        else:
            raise ValueError(
                "The kick velocity model pdf does not exist. Choose between km_maxwell, km_exp or km_2maxwell."
            )

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
        as drawn from a normal or log-normal distribution. The characteristic
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
        Determining the initial dipolar magnetic field strengths of each pulsar in the sample,
        as drawn from a log-normal, double log-normal or smooth top-hat distribution. The characteristic
        parameters are defined in config_simulator.py.

        Returns:
            (np.ndarray): Initial magnetic field strengths of the pulsar sample in [G].
        """

        magnetic_field_model = cfg["magnetic_field_model"]

        if magnetic_field_model == "log-normal":
            B_rand = imf.initial_magnetic_field_lognormal(
                cfg["B_initial_log10_mean"],
                cfg["B_initial_log10_sigma"],
                self.NS_number,
            )
        elif magnetic_field_model == "double_log-normal":
            B_rand = imf.initial_magnetic_field_double_lognormal(
                self.NS_number
            )
        elif magnetic_field_model == "smooth_tophat":
            B_rand = imf.initial_magnetic_field_smooth_tophat(self.NS_number)
        else:
            raise ValueError(
                "The initial magnetic-field model pdf does not exist. Choose between log-normal, double_log-normal or "
                "smooth_tophat."
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
