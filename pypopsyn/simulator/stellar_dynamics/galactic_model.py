"""
Model for the Milky Way gravitational potential.

We consider two different models:

1) gmFK06: A galactic structure as in Faucher-Giguère & Kaspi (2006). Their
model consists of three components: a disk-halo, a bulge and a nucleus.
The parameters of the model are taken from Table B1 in Kuijken & Gilmore (1989).

2) gmM19: The galaxy model from Marchetti et al. (2019). This is a four-component
galactic potential model consisting of a Hernquist bulge and nucleus (Hernquist 1990),
a Miyamoto-Nagai disk (Miyamoto & Nagai 1975) and a Navarro-Frenk-White halo (Navarro
et al. 1996). The parameters of the model are taken from Table 1 in Marchetti et al.
(2019) and are chosen to fit the enclosed mass profile of the Milky Way (Bovy 2015).

To improve performance when evolving the neutron stars' position in the galactic
potential (see dynamical_evolution.py), we add Numba's jit decorator to all functions.

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

import abc
from typing import Tuple

import numpy as np
from numba import float64
from numba.experimental import jitclass

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.configuration import cfg

galactic_model = None


def initialize_galactic_model() -> None:
    """
    Initializing the galactic model employed in the simulation. We have implemented two
    versions, i.e., the galactic structure of Faucher-Giguère & Kaspi (2006) (with
    parameters from Kuijken & Gilmore (1989)) and Galaxy model from Marchetti et al.
    (2019). The galactic_model variable is made available on a global level.
    """
    global galactic_model

    if cfg["galactic_model"] == "gmM19":
        galactic_model = GalaxyModelM19()
    elif cfg["galactic_model"] == "gmFK06":
        galactic_model = GalaxyModelFK06()
    else:
        raise ValueError(
            "The galactic model does not exist. Choose between gmFK06 or gmM19."
        )


class GalaxyModelBase:
    """
    Base class for any galaxy model to ensure that a common interface between all of
    them is respected. The following abstract methods must be implemented or an error
    will be raised.
    """

    @abc.abstractmethod
    def MW_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Please implement the method MW_potential.")

    @abc.abstractmethod
    def cylind_coord_gradient_mw_potential(
        self, r: float, z: float
    ) -> np.ndarray:
        raise NotImplementedError(
            "Please implement the method cylind_coord_gradient_mw_potential."
        )

    def total_energy(
        self, v: np.ndarray, r: np.ndarray, z: np.ndarray
    ) -> float:
        """
        Value of the total energy of the system, sum of the total kinetic energy and
        the total gravitational potential energy. We assume here that all the stars
        have unit mass.

        Args:
            v (np.ndarray): array of magnitudes of the speed of the stars in [km/s].
            r (np.ndarray): array of distances from the galactic axis in [kpc].
            z (np.ndarray): array of distances from the galactic disk in [kpc].

        Returns:
            (float): value of the total energy of the system in [erg].
        """
        # Convert speeds into [cm/s].
        v = v * const.KM_TO_CM

        tot_kin_energy = 0.5 * np.sum(v**2)
        tot_pot_energy = np.sum(self.MW_potential(r, z))

        tot_energy = tot_kin_energy + tot_pot_energy

        return tot_energy

    def total_angular_momentum_z(
        self,
        v_phi: np.ndarray,
        r: np.ndarray,
    ) -> float:
        """
        Value of the z-component of the total angular momentum of the system, which is a
        conserved quantity in axisymmetric potentials. We assume here that all the stars
        have unit mass.

        Args:
            v_phi (np.ndarray): array of magnitudes of the orbital speed of the stars in [km/s].
            r (np.ndarray): array of distances from the galactic axis in [kpc].

        Returns:
            (float): value of the total energy of the system in [erg].
        """
        # Convert speeds into [cm/s].
        v_phi = v_phi * const.KM_TO_CM
        # Convert distances into [cm].
        r = r * const.KPC_TO_CM

        L_z = float(np.sum(r * v_phi))

        return L_z


# Class member data specification for Numba. In order for Numba to be able to
# Jit an entire class, we need to provide a specification of all the data such
# class holds together with its data types. The tuples contain the name of the
# field and the Numba type of the field. By doing so the data of a jitclass
# instance is allocated on the heap as a C-compatible structure so that any
# compiled functions can have direct access to the underlying data, bypassing
# the interpreter.
galaxyModelM19_spec = [
    ("a_d", float64),
    ("b_d", float64),
    ("M_d", float64),
    ("M_b", float64),
    ("r_b", float64),
    ("M_n", float64),
    ("r_n", float64),
    ("M_h", float64),
    ("r_h", float64),
]


@jitclass(galaxyModelM19_spec)
class GalaxyModelM19(GalaxyModelBase):
    """
    Galaxy model from Marchetti et al. (2019). This is a four-component galactic
    potential model consisting of a Hernquist bulge and nucleus (Hernquist 1990),
    a Miyamoto-Nagai disk (Miyamoto & Nagai 1975) and a Navarro-Frenk-White halo
    (Navarro et al. 1996). The parameters of the model are taken from Table 1 in
    Marchetti et al. (2019) and are chosen to fit the enclosed mass profile of
    the Milky Way (Bovy 2015).
    """

    def __init__(self) -> None:
        # Parameters of the model, values from Table 1 in Marchetti et al. (2019).
        self.a_d = 3.0  # Scale length of the disk in [kpc].
        self.b_d = 0.28  # Scale height for the disk.
        self.M_d = 6.8e10 * const.M_SUN  # Disk+halo mass in [g].
        self.M_b = 5.0e9 * const.M_SUN  # Bulge mass in [g].
        self.r_b = 1.0  # Core radius of the bulge component in [kpc].
        self.M_n = 1.71e9 * const.M_SUN  # Nucleus mass in [g].
        self.r_n = 0.07  # Core radius of the nucleus component in [kpc].
        self.M_h = 5.4e11 * const.M_SUN  # Halo mass in [g].
        self.r_h = 15.62  # Core radius of the halo component in [kpc].

    def shape_parameter(self, z: float) -> Tuple[float, float]:
        """
        Shape parameter for the disk potential from Marchetti et al. (2019) and
        its derivative with respect to the height z from the galactic disk.
        Second term in the denominator of eq. (8) in Marchetti et al. (2019).

        Args:
            z (float): height from the galactic disk in [kpc].

        Returns:
            (float, float): value of the shape parameter and its derivative with
            respect to z.
        """

        a_d = self.a_d
        b_d = self.b_d

        K = a_d + np.sqrt(z**2 + b_d**2)

        dK_dz = z / np.sqrt(z**2 + b_d**2)

        return K, dK_dz

    def d_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        The Miyamoto-Nagai disk component gravitational potential defined in eq. (8) in
        Marchetti et al. (2019).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the disk-halo potential in [erg/g].
        """

        K, _ = self.shape_parameter(z)

        M_d = self.M_d

        pot_d = -const.G * M_d / (np.sqrt(K**2 + r**2) * const.KPC_TO_CM)

        return pot_d

    def b_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        The Hernquist bulge component gravitational potential defined in eq. (7) in
        Marchetti et al. (2019).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the bulge potential in [erg/g].
        """
        M_b = self.M_b
        r_b = self.r_b

        pot_b = (
            -const.G
            * M_b
            / ((r_b + np.sqrt(r**2 + z**2)) * const.KPC_TO_CM)
        )

        return pot_b

    def n_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        The Hernquist nucleus component gravitational potential defined in eq. (7) in
        Marchetti et al. (2019).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the bulge potential in [erg/g].
        """
        M_n = self.M_n
        r_n = self.r_n

        pot_n = (
            -const.G
            * M_n
            / ((r_n + np.sqrt(r**2 + z**2)) * const.KPC_TO_CM)
        )

        return pot_n

    def h_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        The Navarro-Frenk-White halo component gravitational potential defined in
        eq. (9) in Marchetti et al. (2019).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the bulge potential in [erg/g].
        """
        M_h = self.M_h
        r_h = self.r_h

        pot_h = (
            -const.G
            * M_h
            / (np.sqrt(r**2 + z**2) * const.KPC_TO_CM)
            * np.log(1.0 + np.sqrt(r**2 + z**2) / r_h)
        )

        return pot_h

    def MW_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Total Milky Way gravitational potential in Marchetti et al. (2019).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the Galactic potential in [erg].
        """

        MW_pot = (
            self.d_potential(r, z)
            + self.b_potential(r)
            + self.n_potential(r)
            + self.h_potential(r)
        )

        return MW_pot

    def r_z_derivatives_d_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r and z of the disk component gravitational
        potential defined in eq. (8) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].

        Returns:
            (float, float): derivative with respect to r and z of the disk potential.
        """

        M_d = self.M_d

        K, dK_dz = self.shape_parameter(z)
        dpot_d_dr = (
            const.G_KPC_YR * M_d * r * (r**2 + K**2) ** (-3.0 / 2.0)
        )
        dpot_d_dz = (
            const.G_KPC_YR
            * M_d
            * (r**2 + K**2) ** (-3.0 / 2.0)
            * K
            * dK_dz
        )

        return dpot_d_dr, dpot_d_dz

    def r_z_derivative_b_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r of the bulge component gravitational potential
        defined in eq. (7) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].
        Returns:
            (float, float): derivative with respect to r and z of the bulge potential.
        """

        M_b = self.M_b
        r_b = self.r_b

        dpot_b_dr = (
            const.G_KPC_YR
            * M_b
            * r
            * ((r_b + np.sqrt(r**2 + z**2)) ** (-2))
            * ((r**2 + z**2) ** (-1 / 2))
        )
        dpot_b_dz = (
            const.G_KPC_YR
            * M_b
            * z
            * ((r_b + np.sqrt(r**2 + z**2)) ** (-2))
            * ((r**2 + z**2) ** (-1 / 2))
        )

        return dpot_b_dr, dpot_b_dz

    def r_z_derivative_n_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r of the nucleus component gravitational potential
        defined in eq. (7) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].
        Returns:
            (float, float): derivative with respect to r and z of the nucleus potential.
        """

        M_n = self.M_n
        r_n = self.r_n

        dpot_n_dr = (
            const.G_KPC_YR
            * M_n
            * r
            * ((r_n + np.sqrt(r**2 + z**2)) ** (-2))
            * ((r**2 + z**2) ** (-1 / 2))
        )
        dpot_n_dz = (
            const.G_KPC_YR
            * M_n
            * z
            * ((r_n + np.sqrt(r**2 + z**2)) ** (-2))
            * ((r**2 + z**2) ** (-1 / 2))
        )

        return dpot_n_dr, dpot_n_dz

    def r_z_derivative_h_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r of the halo component gravitational potential
        defined in eq. (9) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].

        Returns:
            (float, float): derivative with respect to r and z of the halo potential.
        """

        M_h = self.M_h
        r_h = self.r_h

        dpot_h_dr = (const.G_KPC_YR * M_h * r / (r**2 + z**2)) * (
            np.log(1 + np.sqrt(r**2 + z**2) / r_h)
            / (np.sqrt(r**2 + z**2))
            - 1.0 / (r_h + np.sqrt(r**2 + z**2))
        )

        dpot_h_dz = (const.G_KPC_YR * M_h * z / (r**2 + z**2)) * (
            np.log(1 + np.sqrt(r**2 + z**2) / r_h)
            / (np.sqrt(r**2 + z**2))
            - 1.0 / (r_h + np.sqrt(r**2 + z**2))
        )

        return dpot_h_dr, dpot_h_dz

    def cylind_coord_gradient_mw_potential(
        self, r: float, z: float
    ) -> np.ndarray:
        """
        Gradient in cylindrical coordinates of the Milky Way gravitational potential for
        the components defined in eq. (7,8,9) in Marchetti et al. (2019).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].

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


# Class member data specification for Numba. In order for Numba to be able to
# Jit an entire class, we need to provide a specification of all the data such
# class holds together with its data types. The tuples contain the name of the
# field and the Numba type of the field. By doing so the data of a jitclass
# instance is allocated on the heap as a C-compatible structure so that any
# compiled functions can have direct access to the underlying data, bypassing
# the interpreter.
galaxyModelFK06_spec = [
    ("a_d", float64),
    ("h", float64[:]),
    ("beta", float64[:]),
    ("M_dh", float64),
    ("b_dh", float64),
    ("M_b", float64),
    ("b_b", float64),
    ("M_n", float64),
    ("b_n", float64),
]


@jitclass(galaxyModelFK06_spec)
class GalaxyModelFK06(GalaxyModelBase):
    """
    Galaxy model from Faucher-Giguère & Kaspi (2006). This model consists of a
    disk-halo component, a bulge component, and a nucleus component. The
    parameters of the model are taken from Table 1 in Kuijken & Gilmore (1989)
    (in Faucher-Giguère & Kaspi (2006) the nucleus and bulge are erroneously
    inverted).
    """

    def __init__(self) -> None:
        # Parameter values from Table B1 in Kuijken & Gilmore (1989).
        self.a_d = 2.4  # Scale length of the disk in [kpc].
        self.h = np.array(
            [0.325, 0.090, 0.125]
        )  # Array of disk components' scale heights in [kpc].
        self.beta = np.array(
            [0.4, 0.5, 0.1]
        )  # Array of weights for the disk components.
        self.M_dh = 1.45e11 * const.M_SUN  # Disk+halo mass in [g].
        self.b_dh = 5.5  # Core radius of the halo component in [kpc].
        self.M_b = 1.0e10 * const.M_SUN  # Bulge mass in [g].
        self.b_b = 1.5  # Core radius of the bulge component in [kpc].
        self.M_n = 9.3e9 * const.M_SUN  # Nucleus mass in [g].
        self.b_n = 0.25  # Core radius of the nucleus component in [kpc].

    def shape_parameter(self, z: float) -> Tuple[float, float]:
        """
        Shape parameter for the disk-halo potential from Carlberg & Innamen
        (1987) and its derivative with respect to the height z from the galactic
        disk. First term in the denominator of eq. (13) in Faucher-Giguère &
        Kaspi (2006).

        Args:
            z (float): height from the galactic disk in [kpc].

        Returns:
            (float, float): value of the shape parameter and its derivative with
            respect to z.
        """

        a_d = self.a_d
        beta = self.beta
        h = self.h

        K = (
            a_d
            + beta[0] * np.sqrt(z**2 + h[0] ** 2)
            + beta[1] * np.sqrt(z**2 + h[1] ** 2)
            + beta[2] * np.sqrt(z**2 + h[2] ** 2)
        )

        dK_dz = (
            beta[0] * z / np.sqrt(z**2 + h[0] ** 2)
            + beta[1] * z / np.sqrt(z**2 + h[1] ** 2)
            + beta[2] * z / np.sqrt(z**2 + h[2] ** 2)
        )

        return K, dK_dz

    def dh_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        The disk-halo component gravitational potential defined in eq. (14) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the disk-halo potential in [erg/g].
        """

        K, _ = self.shape_parameter(z)

        M_dh = self.M_dh
        b_dh = self.b_dh

        pot_dh = (
            -const.G
            * M_dh
            / (np.sqrt(K**2 + b_dh**2 + r**2) * const.KPC_TO_CM)
        )

        return pot_dh

    def b_potential(self, r: np.ndarray) -> np.ndarray:
        """
        The bulge component gravitational potential defined in eq. (15) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

        Returns:
            (np.ndarray): value of the bulge potential in [erg/g].
        """
        M_b = self.M_b
        b_b = self.b_b

        pot_b = -const.G * M_b / (np.sqrt(b_b**2 + r**2) * const.KPC_TO_CM)

        return pot_b

    def n_potential(self, r: np.ndarray) -> np.ndarray:
        """
        The nucleus component gravitational potential defined in eq. (15) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

        Returns:
            (np.ndarray): value of the nucleus potential.
        """
        M_n = self.M_n
        b_n = self.b_n

        pot_n = -const.G * M_n / (np.sqrt(b_n**2 + r**2) * const.KPC_TO_CM)

        return pot_n

    def MW_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Total Milky Way gravitational potential defined in eq. (13) in
        Faucher-Giguère & Kaspi (2006).

        Args:
            r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
            z (np.ndarray): height from the galactic disk in [kpc].

        Returns:
            (np.ndarray): value of the Galactic potential in [erg].
        """

        MW_pot = (
            self.dh_potential(r, z) + self.b_potential(r) + self.n_potential(r)
        )

        return MW_pot

    def r_z_derivatives_dh_potential(
        self, r: float, z: float
    ) -> Tuple[float, float]:
        """
        Derivative with respect to r and z of the disk-halo component gravitational
        potential defined in eq. (14) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].

        Returns:
            (float, float): derivative with respect to r and z of the disk-halo
            potential.
        """

        M_dh = self.M_dh
        b_dh = self.b_dh

        K, dK_dz = self.shape_parameter(z)
        dpot_dh_dr = (
            const.G_KPC_YR
            * M_dh
            * r
            * (K**2 + b_dh**2 + r**2) ** (-3.0 / 2.0)
        )
        dpot_dh_dz = (
            const.G_KPC_YR
            * M_dh
            * (K**2 + b_dh**2 + r**2) ** (-3.0 / 2.0)
            * K
            * dK_dz
        )

        return dpot_dh_dr, dpot_dh_dz

    def r_derivative_b_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the bulge component gravitational potential
        defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].

        Returns:
            float: derivative with respect to r of the bulge potential.
        """

        M_b = self.M_b
        b_b = self.b_b

        dpot_b_dr = (
            const.G_KPC_YR * M_b * r * (b_b**2 + r**2) ** (-3.0 / 2.0)
        )

        return dpot_b_dr

    def r_derivative_n_potential(self, r: float) -> float:
        """
        Derivative with respect to r of the nucleus component gravitational potential
        defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].

        Returns:
            float: derivative with respect to r of the nucleus potential.
        """

        M_n = self.M_n
        b_n = self.b_n

        dpot_n_dr = (
            const.G_KPC_YR * M_n * r * (b_n**2 + r**2) ** (-3.0 / 2.0)
        )

        return dpot_n_dr

    def cylind_coord_gradient_mw_potential(
        self, r: float, z: float
    ) -> np.ndarray:
        """
        Gradient in cylindrical coordinates of the Milky Way gravitational potential
        defined in eq. (13) in Faucher-Giguère & Kaspi (2006).

        Args:
            r (float): distance in the galactic disk from the galactic centre in [kpc].
            z (float): height from the galactic disk in [kpc].

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
