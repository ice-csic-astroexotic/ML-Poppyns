"""
Model for the Milky Way gravitational potential

We consider two different models:
1) gmFK06: A galactic structure as in Faucher-Giguère & Kaspi (2006). Their
model consists of three components: a disk-halo, a bulge and a nucleus.
The parameters of the model are taken from table B1 in Kuijken & Gilmore (1989).

2) gmM19 The Galaxy model from Marchetti et al. (2019). This is a four components Galactic potential model
consisting of a Hernquist bulge and nucleus (Hernquist 1990), a Miyamoto-Nagai disk (Miyamoto & Nagai 1975)
and a Navarro-Frenk-White halo (Navarro et al. 1996). The parameters of the model are taken from table 1
in Marchetti et al. (2019) and are chosen to fit the enclosed mass profile of the Milky Way (Bovy 2015).

To improve performance when evolving the neutron stars' position in the galactic
potential (see dynamical_evolution.py), we add Numba's jit decorator to all functions.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Tuple

import numpy as np
from numba import jit

import pypopsyn.simulator.constants as const


class GalaxyModelM19:
    """
    Galaxy model from Marchetti et al. (2019). This is a four components Galactic potential model
    consisting of a Hernquist bulge and nucleus (Hernquist 1990), a Miyamoto-Nagai disk (Miyamoto & Nagai 1975)
    and a Navarro-Frenk-White halo (Navarro et al. 1996). The parameters of the model are taken from table 1
    in Marchetti et al. (2019) and are chosen to fit the enclosed mass profile of the Milky Way (Bovy 2015).
    """

    # parameters of the model, values from table 1 in Marchetti et al. (2019).
    a_d = 3.0  # Scale length of the disk in kpc.
    b_d = 0.28  # scale height for the disk.
    M_d = 6.8e10 * const.M_SUN  # Disk+halo mass in g.
    M_b = 5.0e9 * const.M_SUN  # Bulge mass in g.
    r_b = 1.0  # Core radius of the bulge component in kpc.
    M_n = 1.71e9 * const.M_SUN  # nucleus mass in g.
    r_n = 0.07  # Core radius of the nucleus component in kpc.
    M_h = 5.4e11 * const.M_SUN  # halo mass in g.
    r_h = 15.62  # Core radius of the halo component in kpc.

    def shape_parameter(self, z: float) -> Tuple[float, float]:
        """
        Shape parameter for the disk potential from Marchetti et al. (2019) and
        its derivative with respect to the height z from the galactic disk.
        Second term in the denominator of eq. (8) in Marchetti et al. (2019).

        Args:
            z (float): height from the galactic disk in kpc.

        Returns:
            (float, float): value of the shape parameter and its derivative with respect
            to z.
        """

        a_d = self.a_d
        b_d = self.b_d

        K = a_d + np.sqrt(z ** 2 + b_d ** 2)

        dK_dz = z / np.sqrt(z ** 2 + b_d ** 2)

        return K, dK_dz

    def d_potential(self, r: float, z: float) -> float:
        """
        The Miyamoto-Nagai disk component gravitational potential defined in eq. (8) in
        Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float): value of the disk-halo potential in erg/g.
        """

        K, _ = self.shape_parameter(z)

        M_d = self.M_d

        pot_d = -const.G * M_d / (np.sqrt(K ** 2 + r ** 2) * const.KPC_TO_CM)

        return pot_d

    def b_potential(self, r: float) -> float:
        """
        The Hernnquist bulge component gravitational potential defined in eq. (7) in
        Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            (float): value of the bulge potential in erg/g.
        """
        M_b = self.M_b
        r_b = self.r_b

        pot_b = -const.G * M_b / ((r_b + r) * const.KPC_TO_CM)

        return pot_b

    def n_potential(self, r: float) -> float:
        """
        The Hernnquist nucleus component gravitational potential defined in eq. (7) in
        Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            (float): value of the bulge potential in erg/g.
        """
        M_n = self.M_n
        r_n = self.r_n

        pot_n = -const.G * M_n / ((r_n + r) * const.KPC_TO_CM)

        return pot_n

    def h_potential(self, r: float) -> float:
        """
        The Navarro-Frenk-White halo component gravitational potential defined in eq. (9) in
        Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            (float): value of the bulge potential in erg/g.
        """
        M_h = self.M_h
        r_h = self.r_h

        pot_h = -const.G * M_h / (r * const.KPC_TO_CM) * np.log(1.0 + r / r_h)

        return pot_h

    def MW_potential(self, r: float, z: float) -> float:
        """
        Total Milky Way gravitational potential in Marchetti et al. 2019.

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float): value of the Galactic potential in erg.
        """

        MW_pot = (
            self.d_potential(r, z)
            + self.b_potential(r)
            + self.n_potential(r)
            + self.h_potential(r)
        )

        return MW_pot

    def total_energy(
        self, v: np.ndarray, r: np.ndarray, z: np.ndarray
    ) -> float:
        """
        Value of the total energy of the system, sum of the total kinetic energy and the
        total gravitational potential energy. We assume here that all the stars have unit
        mass.

        Args:
            v (np.ndarray): array of magnitudes of the speed of the stars in km/s
            r (np.ndarray): array of distances from the galactic axis in kpc.
            z (np.ndarray): array of distances from the galactic disk in kpc.

        Returns:
            (float): value of the total energy of the system in erg.
        """
        # convert speeds in [cm/s]
        v = v * const.KM_TO_CM

        tot_kin_energy = 0.5 * np.sum(v ** 2)
        tot_pot_energy = np.sum(self.MW_potential(r, z))

        tot_energy = tot_kin_energy + tot_pot_energy

        return tot_energy

    def r_z_derivatives_d_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r and z of the disk component gravitational
        potential defined in eq. (8) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float, float): derivative with respect to r and z of the disk potential.
        """

        M_d = self.M_d

        K, dK_dz = self.shape_parameter(z)
        dpot_d_dr = (
            const.G_KPC_YR * M_d * r * (r ** 2 + K ** 2) ** (-3.0 / 2.0)
        )
        dpot_d_dz = (
            const.G_KPC_YR
            * M_d
            * (r ** 2 + K ** 2) ** (-3.0 / 2.0)
            * K
            * dK_dz
        )

        return dpot_d_dr, dpot_d_dz

    def r_derivative_b_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the bulge component gravitational potential
        defined in eq. (7) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            float: derivative with respect to r of the bulge potential.
        """

        M_b = self.M_b
        r_b = self.r_b

        dpot_b_dr = const.G_KPC_YR * M_b * (r + r_b) ** (-2.0)

        return dpot_b_dr

    def r_derivative_n_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the nucleus component gravitational potential
        defined in eq. (7) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            float: derivative with respect to r of the nucleus potential.
        """

        M_n = self.M_n
        r_n = self.r_n

        dpot_n_dr = const.G_KPC_YR * M_n * (r + r_n) ** (-2.0)

        return dpot_n_dr

    def r_derivative_h_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the halo component gravitational potential
        defined in eq. (9) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            float: derivative with respect to r of the halo potential.
        """

        M_h = self.M_h
        r_h = self.r_h

        dpot_h_dr = (
            const.G_KPC_YR
            * M_h
            / r
            * (1.0 / r * np.log(1 + r / r_h) - 1.0 / (r_h + r))
        )

        return dpot_h_dr

    def cylind_coord_gradient_mw_potential(
        self, r: float, z: float
    ) -> np.ndarray:
        """
        Gradient in cylindrical coordinates of the Milky Way gravitational potential for
        the components defined in eq. (7,8,9) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (np.ndarray): gradient of the galactic potential in cylindrical
            coordinates.
        """

        dpot_d_dr, dpot_d_dz = self.r_z_derivatives_d_potential(r, z)
        dpot_b_dr = self.r_derivative_b_potential(r)
        dpot_n_dr = self.r_derivative_n_potential(r)
        dpot_h_dr = self.r_derivative_h_potential(r)

        dpot_mw_dr = dpot_d_dr + dpot_b_dr + dpot_n_dr + dpot_h_dr
        dpot_mw_dphi = 0.0
        dpot_mw_dz = dpot_d_dz

        pot_mw_gradient = np.array([dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz])

        return pot_mw_gradient


class GalaxyModelFK06:
    """
    Galaxy model from Faucher-Giguère & Kaspi (2006). This model consists ofa disk-halo component, a bulge component,
    and a nucleus component. The parameters of the model are taken from table 1 in Kuijken & Gilmore (1989)
    (in Faucher-Giguère & Kaspi 2006 the nucleus and bulge are erroneously inverted).
    """

    # Parameter values from table B1 in Kuijken & Gilmore (1989)
    a_d = 2.4  # Scale length of the disk in kpc.
    h = np.array(
        [0.325, 0.090, 0.125]
    )  # Array of disk components' scale heights in kpc.
    beta = np.array(
        [0.4, 0.5, 0.1]
    )  # Array of weights for the disk components.
    M_dh = 1.45e11 * const.M_SUN  # Disk+halo mass in g.
    b_dh = 5.5  # Core radius of the halo component in kpc.
    M_b = 1.0e10 * const.M_SUN  # Bulge mass in g.
    b_b = 1.5  # Core radius of the bulge component in kpc.
    M_n = 9.3e9 * const.M_SUN  # Nucleus mass in g.
    b_n = 0.25  # Core radius of the nucleus component in kpc.

    def shape_parameter(self, z: float) -> Tuple[float, float]:
        """
        Shape parameter for the disk-halo potential from Carlberg & Innamen (1987) and
        its derivative with respect to the height z from the galactic disk.
        First term in the denominator of eq. (13) in Faucher-Giguère & Kaspi (2006).

        Args:
            z (float): height from the galactic disk in kpc.

        Returns:
            (float, float): value of the shape parameter and its derivative with respect
            to z.
        """

        a_d = self.a_d
        beta = self.beta
        h = self.h

        K = (
            a_d
            + beta[0] * np.sqrt(z ** 2 + h[0] ** 2)
            + beta[1] * np.sqrt(z ** 2 + h[1] ** 2)
            + beta[2] * np.sqrt(z ** 2 + h[2] ** 2)
        )

        dK_dz = (
            beta[0] * z / np.sqrt(z ** 2 + h[0] ** 2)
            + beta[1] * z / np.sqrt(z ** 2 + h[1] ** 2)
            + beta[2] * z / np.sqrt(z ** 2 + h[2] ** 2)
        )

        return K, dK_dz

    def dh_potential(self, r: float, z: float) -> float:
        """
        The disk-halo component gravitational potential defined in eq. (14) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float): value of the disk-halo potential in erg/g.
        """

        K, _ = self.shape_parameter(z)

        M_dh = self.M_dh
        b_dh = self.b_dh

        pot_dh = (
            -const.G
            * M_dh
            / (np.sqrt(K ** 2 + b_dh ** 2 + r ** 2) * const.KPC_TO_CM)
        )

        return pot_dh

    def b_potential(self, r: float) -> float:
        """
        The bulge component gravitational potential defined in eq. (15) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            (float): value of the bulge potential in erg/g.
        """
        M_b = self.M_b
        b_b = self.b_b

        pot_b = -const.G * M_b / (np.sqrt(b_b ** 2 + r ** 2) * const.KPC_TO_CM)

        return pot_b

    def n_potential(self, r: float) -> float:
        """
        The nucleus component gravitational potential defined in eq. (15) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            (float): value of the nucleus potential.
        """
        M_n = self.M_n
        b_n = self.b_n

        pot_n = -const.G * M_n / (np.sqrt(b_n ** 2 + r ** 2) * const.KPC_TO_CM)

        return pot_n

    def MW_potential(self, r: float, z: float) -> float:
        """
        Total Milky Way gravitational potential defined in eq. (13) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float): value of the Galactic potential in erg.
        """

        MW_pot = (
            self.dh_potential(r, z) + self.b_potential(r) + self.n_potential(r)
        )

        return MW_pot

    def total_energy(
        self, v: np.ndarray, r: np.ndarray, z: np.ndarray
    ) -> float:
        """
        Value of the total energy of the system, sum of the total kinetic energy and the
        total gravitational potential energy. We assume here that all the stars have unit
        mass.

        Args:
            v (np.ndarray): array of magnitudes of the speed of the stars in km/s
            r (np.ndarray): array of distances from the galactic axis in kpc.
            z (np.ndarray): array of distances from the galactic disk in kpc.

        Returns:
            (float): value of the total energy of the system in erg.
        """
        # convert speeds in [cm/s]
        v = v * const.KM_TO_CM

        tot_kin_energy = 0.5 * np.sum(v ** 2)
        tot_pot_energy = np.sum(self.MW_potential(r, z))

        tot_energy = tot_kin_energy + tot_pot_energy

        return tot_energy

    def r_z_derivatives_dh_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r and z of the disk-halo component gravitational
        potential defined in eq. (14) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (float, float): derivative with respect to r and z of the disk-halo potential.
        """

        M_dh = self.M_dh
        b_dh = self.b_dh

        K, dK_dz = self.shape_parameter(z)
        dpot_dh_dr = (
            const.G_KPC_YR
            * M_dh
            * r
            * (K ** 2 + b_dh ** 2 + r ** 2) ** (-3.0 / 2.0)
        )
        dpot_dh_dz = (
            const.G_KPC_YR
            * M_dh
            * (K ** 2 + b_dh ** 2 + r ** 2) ** (-3.0 / 2.0)
            * K
            * dK_dz
        )

        return dpot_dh_dr, dpot_dh_dz

    def r_derivative_b_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the bulge component gravitational potential
        defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            float: derivative with respect to r of the bulge potential.
        """

        M_b = self.M_b
        b_b = self.b_b

        dpot_b_dr = (
            const.G_KPC_YR * M_b * r * (b_b ** 2 + r ** 2) ** (-3.0 / 2.0)
        )

        return dpot_b_dr

    def r_derivative_n_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the nucleus component gravitational potential
        defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.

        Returns:
            float: derivative with respect to r of the nucleus potential.
        """

        M_n = self.M_n
        b_n = self.b_n

        dpot_n_dr = (
            const.G_KPC_YR * M_n * r * (b_n ** 2 + r ** 2) ** (-3.0 / 2.0)
        )

        return dpot_n_dr

    def cylind_coord_gradient_mw_potential(
        self, r: float, z: float
    ) -> np.ndarray:
        """
        Gradient in cylindrical coordinates of the Milky Way gravitational potential
        defined in eq. (13) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in kpc.
            z (float): height from the galactic disk in kpc.

        Returns:
            (np.ndarray): gradient of the galactic potential in cylindrical
            coordinates.
        """

        dpot_dh_dr, dpot_dh_dz = self.r_z_derivatives_dh_potential(r, z)
        dpot_b_dr = self.r_derivative_b_potential(r)
        dpot_n_dr = self.r_derivative_n_potential(r)

        dpot_mw_dr = dpot_dh_dr + dpot_b_dr + dpot_n_dr
        dpot_mw_dphi = 0.0
        dpot_mw_dz = dpot_dh_dz

        pot_mw_gradient = np.array([dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz])

        return pot_mw_gradient
