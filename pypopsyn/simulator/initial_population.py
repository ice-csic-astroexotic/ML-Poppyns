"""
Generating an initial population of neutron stars in the Milky Way with random parameters.
"""

from typing import Tuple

import numpy as np

import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.initial_position as ip
import pypopsyn.simulator.initial_velocity as iv


class InitialNeutronStarPopulation:
    """
    Generating a random pulsar population in the Milky Way
    """

    def __init__(
        self,
        r_extent=20.0,  # [kpc]
        z_extent=5.0,  # [kpc]
        resolution=10000,
        NS_number=50000,
        arm_number=4,
    ):
        """
        Predefined values for the simulation

        Args:
            r_extent (float): total radial extent from the galactic centre in kpc
            z_extent (float): total vertical extent from the galactic plane in kpc
            resolution (int): spatial resolution of the simulation grid
            NS_number (int): total number of neutron stars created
            arm_number (int): number of spiral arms in the galaxy
        """

        self.r_extent = r_extent
        self.z_extent = z_extent
        self.resolution = resolution
        self.NS_number = NS_number
        self.arm_number = arm_number

    def position(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculating the position of each random neutron star in Cartesian coordinates.

        Returns:
            (np.nbarray, np.nbarray, np.nbarray): x, y and z coordinate in kpc for each
            generated neutron stars
        """

        # calculating the cumulative distribution function for the stellar density
        # distribution in the galactic plane
        r_grid = np.logspace(
            np.log10(0.0001), np.log10(self.r_extent), self.resolution
        )
        cdf_rho = cc.cdf_calculator(r_grid, ip.stellar_surf_density)

        # uniformly drawing a random number of NS_number from the cdf
        # to produce random distances in kpc for each of the neutron stars
        cdf_rho_rand = np.random.uniform(0, 1, self.NS_number)
        r_cdf_rand = np.interp(cdf_rho_rand, cdf_rho, r_grid)

        # randomly select one of the four spiral arms for the neutron star sample
        arm_index_rand = np.random.randint(
            1, self.arm_number + 1, self.NS_number
        )

        # evaluate the angular theta coordinate for each neutron star and add noise
        # to both galactocentric coordinates
        theta_rand = np.zeros(self.NS_number)
        r_rand = np.zeros(self.NS_number)
        for i in range(self.NS_number):
            theta_rand[i], r_rand[i] = ip.pdf_initial_coordinates(
                r_cdf_rand[i], arm_index_rand[i]
            )

        # position in the galactic plane in Cartesian coordinates
        x_rand = r_rand * np.cos(theta_rand)
        y_rand = r_rand * np.sin(theta_rand)

        # calculating the cumulative distribution function for the height of the
        # stellar distribution
        z_grid = np.logspace(
            np.log10(0.0001), np.log10(self.z_extent), self.resolution
        )
        cdf_height = cc.cdf_calculator(z_grid, ip.pdf_initial_height)

        # uniformly drawing a random number of NS_number from the cdf
        # to produce random height in kpc for each of the neutron stars
        cdf_height_rand = np.random.uniform(0, 1, self.NS_number)
        z_cdf_rand = np.interp(cdf_height_rand, cdf_height, z_grid)

        # randomly distribute the stars above and below the galactic plane
        up_down_index = np.random.randint(0, 2, self.NS_number)
        z_rand = np.zeros(self.NS_number)
        for i in range(self.NS_number):
            if up_down_index[i] == 0:
                z_rand[i] = -z_cdf_rand[i]
            else:
                z_rand[i] = z_cdf_rand[i]

        return x_rand, y_rand, z_rand
