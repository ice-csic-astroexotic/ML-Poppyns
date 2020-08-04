"""
Generating an initial population of neutron stars in the Milky Way with
random parameters.

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
import time
from typing import Tuple

import numpy as np

import pypopsyn.benchmark.pyinstrument as benchmark
import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.constants as const
import pypopsyn.simulator.coordinate_conversions as coco
import pypopsyn.simulator.initial_position as ip
import pypopsyn.simulator.initial_velocity as iv
from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way.
    """

    def __init__(self) -> None:
        """
        Initialization for the initial population synthesis.
        """

        # Initialize seed randomly if no seed was specified.
        if cfg["seed"] is None:
            cfg["seed"] = int(time.time())

        # Set NumPy random set globally.
        log.info("Seed: {}".format(cfg["seed"]))
        np.random.seed(cfg["seed"])

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
            cfg["t_age_min"], cfg["t_age_max"], cfg["NS_number"]
        )
        return t_age

    @benchmark.profile(
        enabled=cfg["enable_profiles"],
        show=cfg["show_profiles"],
        output_dir=cfg["profiles_dir"],
    )
    def position(
        self, t_age: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in
        cylindrical and Cartesian coordinates in a galactocentric reference frame.

        Args:
            t_age (np.ndarray): array of neutron star ages in [yr].

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            polar r and phi coordinates in [kpc] and rad and Cartesian x, y and z
            coordinates in [kpc] for each generated neutron star.
        """

        # Drawing a random distance from the galactic center in [kpc] for
        # each neutron star according to the radial stellar density.
        r_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["r_extent"]), cfg["resolution"]
        )
        r_pdf_rand = cc.random_from_pdf(
            r_grid, ip.pdf_radial_stellar_density, cfg["NS_number"],
        )

        # Randomly select one of the four spiral arms for the neutron star sample.
        arm_index_rand = np.random.randint(
            1, cfg["arm_number"] + 1, cfg["NS_number"]
        )

        # Evaluate the angular phi coordinate for each neutron star and
        # add noise to both galactocentric coordinates.
        phi_rand = np.zeros(cfg["NS_number"])
        r_rand = np.zeros(cfg["NS_number"])
        phi_rand, r_rand = ip.pdf_initial_coordinates(
            r_pdf_rand, cfg["NS_number"], arm_index_rand
        )

        # Propagating the azimuthal coordinate of each object backwards in time
        # (according to its age) to account for the rotation of the galactic arms;
        # we assume that the arm structure itself remains rigid.
        phi_rand = ip.spiral_arm_time_evol(phi_rand, t_age)

        # Position in the galactic plane in Cartesian coordinates.
        polar_to_cartesian_vect = np.vectorize(coco.polar_to_cartesian)
        x_rand, y_rand = polar_to_cartesian_vect(r_rand, phi_rand)

        # Drawing a random distance from the galactic plane in [kpc] for each neutron
        # star according to the probability density function for the height.
        z_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["z_extent"]), cfg["resolution"]
        )
        z_pdf_rand = cc.random_from_pdf(
            z_grid, ip.pdf_initial_height, cfg["NS_number"]
        )

        # Randomly distribute the stars above and below the galactic plane.
        z_rand = ip.random_scatter_about_plane(z_pdf_rand, cfg["NS_number"])

        return r_rand, phi_rand, x_rand, y_rand, z_rand

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
        else:
            raise ValueError(
                "The kick velocity model pdf does not exist. Choose between km_maxwell or km_exp."
            )

        # Drawing a random magnitude of the birth kick velocity in [km/s] for each
        # neutron star according to the underlying velocity probability density
        # function.
        vk_grid = np.linspace(0.0, cfg["vk_extent"], cfg["resolution"])
        vk_rand = cc.random_from_pdf(vk_grid, pdf_vkick, cfg["NS_number"])
        # Convert from [km/s] to [kpc/yr].
        vk_rand = vk_rand * const.YR_TO_S / const.KPC_TO_KM

        # To draw a random direction for the speed from a uniform distribution,
        # we uniformly sample the azimuthal angle [rad] in the range [0, 2*np.pi];
        # to obtain a uniform distribution in the z-direction, we sample the polar
        # angle [rad] in the range [0, np.pi] according to the PDF np.sin.
        psi_rand = np.random.uniform(0, 2 * np.pi, cfg["NS_number"])
        theta_grid = np.linspace(0.0, np.pi, cfg["resolution"])
        theta_rand = cc.random_from_pdf(theta_grid, np.sin, cfg["NS_number"])

        # Project the velocity on a Cartesian reference frame co-moving with each
        # star, where the local x-axis points always in the r-direction of our
        # galactocentric frame, the local y-axis in the azimuthal phi-direction
        # and the local z-axis coincides with the galactocentric one.
        spherical_to_cartesian_vect = np.vectorize(coco.spherical_to_cartesian)
        vk_r_rand, vk_phi_rand, vk_z_rand = spherical_to_cartesian_vect(
            vk_rand, theta_rand, psi_rand
        )

        return vk_r_rand, vk_phi_rand, vk_z_rand

    @staticmethod
    def orbital_velocity(
        r: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
