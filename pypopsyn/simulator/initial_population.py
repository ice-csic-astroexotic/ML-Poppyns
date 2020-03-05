"""
Generating an initial population of neutron stars in the Milky Way with random parameters.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Tuple

import numpy as np

import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.constants as const
import pypopsyn.simulator.coordinate_conversions as coco
import pypopsyn.simulator.initial_position as ip
import pypopsyn.simulator.initial_velocity as iv
from pypopsyn.simulator.configuration import cfg


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way
    """

    def __init__(self, seed=None):
        """
        Initialization for the population synthesis.

        Args:
            seed (int): seed for random number generation,
                        set to None unless otherwise specified
        """

        self.seed = seed

        np.random.seed(seed)

    def age(self) -> np.ndarray:
        """
        Drawing a random age in years for each neutron star from a uniform probability
        distribution in a given range of time.

        Returns:
            np.array : array of ages in years
        """

        t_age = np.random.uniform(
            cfg["t_age_min"], cfg["t_age_max"], cfg["NS_number"]
        )
        return t_age

    def position(
        self, t_age: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in
        cylindrical and Cartesian coordinates in a galactocentric reference frame.

        Args:
            t_age (np.ndarray): array of ages in years of each neutron star

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            polar r and phi coordinates in kpc and rad and Cartesian x, y and z
            coordinates in kpc for each generated neutron star
        """

        # drawing a random distance from the galactic center in kpc for each neutron
        # star according to the radial stellar density
        r_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["r_extent"]), cfg["resolution"]
        )
        r_pdf_rand = cc.random_from_pdf(
            r_grid, ip.pdf_radial_stellar_density, cfg["NS_number"]
        )

        # Randomly select one of the four spiral arms for the neutron star sample.
        arm_index_rand = np.random.randint(
            1, cfg["arm_number"] + 1, cfg["NS_number"]
        )

        # Evaluate the angular phi coordinate for each neutron star and add noise
        # to both galactocentric coordinates.
        phi_rand = np.zeros(cfg["NS_number"])
        r_rand = np.zeros(cfg["NS_number"])
        for i in range(cfg["NS_number"]):
            phi_rand[i], r_rand[i] = ip.pdf_initial_coordinates(
                r_pdf_rand[i], arm_index_rand[i]
            )

            phi_rand[i] = ip.spiral_arm_time_evol(phi_rand[i], t_age[i])

        # position in the galactic plane in Cartesian coordinates
        polar_to_cartesian_vect = np.vectorize(coco.polar_to_cartesian)
        x_rand, y_rand = polar_to_cartesian_vect(r_rand, phi_rand)

        # drawing a random height from the galactic plane in kpc for each neutron
        # star according to the height stellar density
        z_grid = np.logspace(
            np.log10(0.0001), np.log10(cfg["z_extent"]), cfg["resolution"]
        )
        z_pdf_rand = cc.random_from_pdf(
            z_grid, ip.pdf_initial_height, cfg["NS_number"]
        )

        # randomly distribute the stars above and below the galactic plane
        z_rand = ip.random_scatter_about_plane(z_pdf_rand, cfg["NS_number"])

        return r_rand, phi_rand, x_rand, y_rand, z_rand

    def proper_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the proper velocity of each random neutron star in a cylindrical
        galactocentric coordinate system.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): vp_r, vp_phi and vp_z proper
            velocities in kpc / yr for each generated neutron stars. In particular
            vp_r is the component of the proper velocity along the galactocentric
            radial direction, vp_phi is the component along the azimuthal phi
            direction and vp_z is the component along the z direction.
        """

        # drawing a random magnitude of the proper velocity in km / s for each neutron
        # star according to the proper velocity probability distribution
        vp_grid = np.linspace(0.0, cfg["vp_extent"], cfg["resolution"])
        vp_rand = cc.random_from_pdf(
            vp_grid, iv.pdf_proper_velocity, cfg["NS_number"]
        )
        # convert from km / s to kpc / yr
        vp_rand = vp_rand * const.YR_TO_S / const.KPC_TO_KM

        # drawing a random direction for the speed
        # drawing a random polar angle [0,np.pi] [rad]
        theta_grid = np.linspace(0.0, np.pi, cfg["resolution"])
        theta_rand = cc.random_from_pdf(theta_grid, np.sin, cfg["NS_number"])

        # drawing a random psi angle [0,2np.pi] [rad]
        psi_rand = np.random.uniform(0, 2 * np.pi, cfg["NS_number"])

        # project the velocity on a cartesian reference frame comoving with the star
        # where the x axis points always in the r direction, the y axis in the
        # azimuthal phi direction and the z axes coincide
        spherical_to_cartesian_vect = np.vectorize(coco.spherical_to_cartesian)
        vp_r_rand, vp_phi_rand, vp_z_rand = spherical_to_cartesian_vect(
            vp_rand, theta_rand, psi_rand
        )

        return vp_r_rand, vp_phi_rand, vp_z_rand

    @staticmethod
    def orbital_velocity(
        r: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the orbital circular velocity of each star in the galactic
        gravitational potential. In galactocentric cylindrical coordinates,
        the only non-zero component is the azimuthal phi component. Since the stars
        in the galaxy rotate in the clockwise direction, i.e towards decreasing phi
        values, the phi component is negative.

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre
            in kpc
            z (np.ndarray): height from the galactic disk in kpc

        Returns:
            (np.ndarray): array of orbital velocities in kpc / yr
        """
        virial_orbital_velocity_vect = np.vectorize(iv.virial_orbital_velocity)
        v_orb = -virial_orbital_velocity_vect(r, z)

        return v_orb
