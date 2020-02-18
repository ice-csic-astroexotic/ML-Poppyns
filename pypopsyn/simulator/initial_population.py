"""
Generating an initial population of neutron stars in the Milky Way with random parameters.
"""

from typing import Tuple

import numpy as np

import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.coordinate_conversions as coco
import pypopsyn.simulator.initial_position as ip
import pypopsyn.simulator.initial_velocity as iv

# unit convertions
kpc_to_km = 3.08567758e16  # convert from kpc to km
yr_to_s = 3600 * 24 * 365  # convert from yr to s


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way
    """

    def __init__(
        self,
        r_extent=20.0,  # [kpc]
        z_extent=5.0,  # [kpc]
        vp_extent=2000.0,  # [km s^(-1)]
        t_age_range=np.array([1.0, 1.0e9]),  # [years]
        resolution=10000,
        NS_number=50000,
        arm_number=4,
        seed=None,
    ):
        """
        Predefined values for the simulation

        Args:
            r_extent (float): total radial extent from the galactic centre in kpc
            z_extent (float): total vertical extent from the galactic plane in kpc
            vp_extent (float): maximum proper velocity magnitude in km / s
            t_age_range (float): range of neutron stars age in years
            resolution (int): spatial resolution of the simulation grid
            NS_number (int): total number of neutron stars created
            arm_number (int): number of spiral arms in the galaxy
            seed (int): seed for random number generation,
                        set to None unless otherwise specified
        """

        self.r_extent = r_extent
        self.z_extent = z_extent
        self.vp_extent = vp_extent
        self.t_age_range = t_age_range
        self.resolution = resolution
        self.NS_number = NS_number
        self.arm_number = arm_number
        self.seed = seed

        np.random.seed(seed)

    def position(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position at birth of each random neutron star in Cartesian
        coordinates.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): x, y and z coordinate in kpc for each
            generated neutron stars
        """

        # drawing a random age for each neutron star from a uniform distribution
        t_age = np.random.uniform(
            self.t_age_range[0], self.t_age_range[1], self.NS_number
        )

        # drawing a random distance from the galactic center in kpc for each neutron
        # star according to the radial stellar density
        r_grid = np.logspace(
            np.log10(0.0001), np.log10(self.r_extent), self.resolution
        )
        r_pdf_rand = cc.random_from_pdf(
            r_grid, ip.pdf_radial_stellar_density, self.NS_number
        )

        # randomly select one of the four spiral arms for the neutron star sample
        arm_index_rand = np.random.randint(
            1, self.arm_number + 1, self.NS_number
        )

        # evaluate the angular phi coordinate for each neutron star taking into
        # account that at its birth the arm was in a different position due to the
        # spiral pattern rotation and add noise to both galactocentric coordinates
        phi_rand = np.zeros(self.NS_number)
        r_rand = np.zeros(self.NS_number)
        for i in range(self.NS_number):
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
            np.log10(0.0001), np.log10(self.z_extent), self.resolution
        )
        z_pdf_rand = cc.random_from_pdf(
            z_grid, ip.pdf_initial_height, self.NS_number
        )

        # randomly distribute the stars above and below the galactic plane
        z_rand = ip.random_scatter_about_plane(z_pdf_rand, self.NS_number)

        return x_rand, y_rand, z_rand

    def cartesian_proper_velocity(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the proper velocity of each random neutron star in Cartesian coordinates.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): vp_x, vp_y and vp_z proper
            velocities in kpc / yr for each generated neutron stars
        """

        # drawing a random magnitude of the proper velocity in km / s for each neutron
        # star according to the proper velocity probability distribution
        vp_grid = np.linspace(0.0, self.vp_extent, self.resolution)
        vp_rand = cc.random_from_pdf(
            vp_grid, iv.pdf_proper_velocity, self.NS_number
        )
        # convert from km / s to kpc / yr
        vp_rand = vp_rand * yr_to_s / kpc_to_km

        # drawing a random direction for the speed
        # drawing a random polar angle [0,np.pi] [rad]
        theta_grid = np.linspace(0.0, np.pi, self.resolution)
        theta_rand = cc.random_from_pdf(theta_grid, np.sin, self.NS_number)

        # drawing a random psi angle [0,2np.pi] [rad]
        psi_rand = np.random.uniform(0, 2 * np.pi, self.NS_number)

        # project the velocity on the cartesian axes
        spherical_to_cartesian_vect = np.vectorize(coco.spherical_to_cartesian)
        vp_x_rand, vp_y_rand, vp_z_rand = spherical_to_cartesian_vect(
            vp_rand, theta_rand, psi_rand
        )

        return vp_x_rand, vp_y_rand, vp_z_rand

    def cylindrical_proper_velocity(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        vp_grid = np.linspace(0.0, self.vp_extent, self.resolution)
        vp_rand = cc.random_from_pdf(
            vp_grid, iv.pdf_proper_velocity, self.NS_number
        )
        # convert from km / s to kpc / yr
        vp_rand = vp_rand * yr_to_s / kpc_to_km

        # drawing a random direction for the speed
        # drawing a random polar angle [0,np.pi] [rad]
        theta_grid = np.linspace(0.0, np.pi, self.resolution)
        theta_rand = cc.random_from_pdf(theta_grid, np.sin, self.NS_number)

        # drawing a random psi angle [0,2np.pi] [rad]
        psi_rand = np.random.uniform(0, 2 * np.pi, self.NS_number)

        # project the velocity on a cartesian reference frame comoving with the star
        # where the x axis points always in the r direction, the y axis in the
        # azimuthal phi direction and the z axes coincide
        spherical_to_cartesian_vect = np.vectorize(coco.spherical_to_cartesian)
        vp_r_rand, vp_phi_rand, vp_z_rand = spherical_to_cartesian_vect(
            vp_rand, theta_rand, psi_rand
        )

        return vp_r_rand, vp_phi_rand, vp_z_rand
