"""
    This module compute the evolution in time of the pulsar dipolar magnetic field.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import math

import numpy as np
from numba import float64, jit

import mlpoppyns.simulator.basics.constants as const


@jit(float64(float64, float64))
def timescale_ohmic(L: float, sigma: float) -> float:
    """
    Calculating the ohmic diffusion timescale for a given conductivity and characteristic magnetic
    field length scale. Note that for our purposes, we neglect the fact that both quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the ohmic diffusion process. For our choices see the configuration file.

    Args:
        L (float): Characteristic length scale on which the magnetic field varies, measured in [cm].
        sigma (float): Conductivity of the dominating dissipative process, measure in [1/s].

    Returns:
        (float): Ohmic diffusion timescale in [yr].
    """

    tau_ohm = 4 * np.pi * sigma * L**2 / (const.C**2 * const.YR_TO_S)

    return tau_ohm


@jit(float64(float64, float64, float64))
def timescale_Hall(B: float, L: float, n_e: float) -> float:
    """
    Calculating the Hall timescale for a given field strength, characteristic magnetic field length
    scale and electron density. Note that for our purposes, we neglect the fact that all quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the conservative Hall process. For our choices of L and n_e see the configuration file.
    B will be identified with the initial dipolar magnetic field components at the pulsars' pole.

    Args:
        B (float): Local magnetic field strength, measured in [G].
        L (float): Characteristic length scale on which the magnetic field varies, measured in [cm].
        n_e (float): Electron density, measured in [g/cm^3].

    Returns:
        (float): Hall timescale in [yr].
    """

    tau_Hall = (
        4 * np.pi * const.E * n_e * L**2 / (const.C * B * const.YR_TO_S)
    )

    return tau_Hall


def magnetic_field_evolution_analytical_numpy(
    B_initial: float,
    t: np.ndarray,
    B_asymptotic: float,
    L: float,
    sigma: float,
    n_e: float,
) -> np.ndarray:
    """
    Calculating the evolution of the magnetic field strength of a pulsar based on a simplified model (see eq. (17)
    of Aguilera et al. (2008)) that captures the characteristics of more complicated numerical simulations of pulsar
    magnetic field evolution, i.e., at early timescales the Hall evolution dominates, while at late times the
    exponential magnetic field decay due to Ohmic dissipation kicks in. Note that as explained in Aguilera et al.
    (2008) the Hall timescale corresponds to that of the initial field strength.

    This method is compatible with NumPy arrays and is used if one wants to save the entire evolution output in the
    magneto_rotational_evolution method.

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (np.ndarray): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        L (float): Characteristic length scale on which the magnetic field varies, measured in [cm].
        sigma (float): Conductivity of the dominating dissipative process, measure in [1/s].
        n_e (float): Electron density, measured in [g/cm^3].

    Returns:
        (float): Magnetic field derivatives for a simulated pulsars in [G/yr].
    """

    tau_ohm = timescale_ohmic(L, sigma)
    tau_hall = timescale_Hall(B_initial, L, n_e)

    B = (
        B_initial
        * np.exp(-t / tau_ohm)
        / (1.0 + tau_ohm / tau_hall * (1.0 - np.exp(-t / tau_ohm)))
    )

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = np.ones(len(t)) * B_initial
    elif B_initial > B_asymptotic:
        B[B < B_asymptotic] = B_asymptotic

    return B


@jit(float64(float64, float64, float64, float64, float64, float64))
def magnetic_field_evolution_analytical(
    B_initial: float,
    t: float,
    B_asymptotic: float,
    L: float,
    sigma: float,
    n_e: float,
) -> float:
    """
    Calculating the evolution of the magnetic field strength of a pulsar based on a simplified model (see eq. (17)
    of Aguilera et al. (2008)) that captures the characteristics of more complicated numerical simulations of pulsar
    magnetic field evolution, i.e., at early timescales the Hall evolution dominates, while at late times the
    exponential magnetic field decay due to Ohmic dissipation kicks in. Note that as explained in Aguilera et al.
    (2008) the Hall timescale corresponds to that of the initial field strength.

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (float): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        L (float): Characteristic length scale on which the magnetic field varies, measured in [cm].
        sigma (float): Conductivity of the dominating dissipative process, measure in [1/s].
        n_e (float): Electron density, measured in [g/cm^3].

    Returns:
        (float): Magnetic field derivatives for a simulated pulsars in [G/yr].
    """

    tau_ohm = timescale_ohmic(L, sigma)
    tau_hall = timescale_Hall(B_initial, L, n_e)

    B = (
        B_initial
        * math.exp(-t / tau_ohm)
        / (1.0 + tau_ohm / tau_hall * (1.0 - math.exp(-t / tau_ohm)))
    )

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = B_initial
    elif B < B_asymptotic:
        B = B_asymptotic

    return B


def magnetic_field_evolution_fit_numpy(
    B_initial: float,
    t: np.ndarray,
    B_asymptotic: float,
    a1: float,
    a2: float,
    A1: float,
    A2: float,
    b1: float,
    b2: float,
    tau_late: float,
    a_late: float,
) -> np.ndarray:
    """
    An analytical function for the magnetic field evolution curves from the magneto-thermal evolution simulations.
    This method is compatible with NumPy arrays and is used if one wants to save the entire evolution output in the
    magneto_rotational_evolution method.

    The fit parameters for each magneto-thermal model specified in the config_simulator.py file were adjusted by hand
    (see the notebook tutorials/analysis_notebooks/magnetic_field_evolution_fit.ipynb for more details).

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (np.ndarray): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        a1 (float): Power-law index for the first power-law component for the magnetic-field evolution fit.
        a2 (float): Power-law index for the second power-law component for the magnetic-field evolution fit.
        A1 (float): Normalization for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        A2 (float): Normalization for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        b1 (float): Power-law index for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        b2 (float): Power-law index for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        tau_late (float): Timescale in [yr] when transitioning from the simulated curves to the simple late-time
            power-law evolution of the magnetic field strength.
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.

    Returns:
        (np.ndarray): Magnetic field evolution in [G] as a function of time t.
    """
    # We consider an evolution determined by three main timescales tau1, tau2 and tau_late.
    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code. At late times after a timescale tau_late we assume that the evolution is determined by a simple power-law.

    B = np.zeros(len(t))

    # Define the two timescales as a function of the initial B field.
    tau1 = A1 * B_initial**b1
    tau2 = A2 * B_initial**b2

    if tau2 < tau_late:
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau2) ** (a2 - a1)
            * (1 + t / tau_late) ** (a_late - a2)
        )

    elif (tau1 < tau_late) & (tau_late < tau2):
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau_late) ** (a_late - a1)
        )
    elif tau_late < tau1:
        B = B_initial * (1 + t / tau_late) ** a_late

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = np.ones(len(t)) * B_initial
    elif B_initial > B_asymptotic:
        B[B < B_asymptotic] = B_asymptotic

    return B


@jit(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def magnetic_field_evolution_fit(
    B_initial: float,
    t: float,
    B_asymptotic: float,
    a1: float,
    a2: float,
    A1: float,
    A2: float,
    b1: float,
    b2: float,
    tau_late: float,
    a_late: float,
) -> float:
    """
    An analytical fit for the magnetic field evolution curves from the magneto-thermal evolution simulations.
    This method is used when solving the differential equations if one wants to save only the final state in the
    magneto_rotational_evolution method.

    The fit parameters for each magneto-thermal model specified in the config_simulator.py file were adjusted by hand
    (see the notebook tutorials/analysis_notebooks/magnetic_field_evolution_fit.ipynb for more details).

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (float): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        a1 (float): Power-law index for the first power-law component for the magnetic-field evolution fit.
        a2 (float): Power-law index for the second power-law component for the magnetic-field evolution fit.
        A1 (float): Normalization for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        A2 (float): Normalization for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        b1 (float): Power-law index for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        b2 (float): Power-law index for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        tau_late (float): Timescale in [yr] when transitioning from the simulated curves to the simple late-time
            power-law evolution of the magnetic field strength.
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.

    Returns:
        (float): Magnetic field value in [G] at time t.
    """
    # We consider an evolution determined by three main timescales tau1, tau2 and tau_late.
    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code. At late times after a timescale tau_late we assume that the evolution is determined by a simple power-law.

    B: float = B_initial

    # Define the two timescales as a function of the initial B field.
    tau1 = A1 * B_initial**b1
    tau2 = A2 * B_initial**b2

    if tau2 < tau_late:
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau2) ** (a2 - a1)
            * (1 + t / tau_late) ** (a_late - a2)
        )

    elif (tau1 < tau_late) & (tau_late < tau2):
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau_late) ** (a_late - a1)
        )
    elif tau_late < tau1:
        B = B_initial * (1 + t / tau_late) ** a_late

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = B_initial
    elif B < B_asymptotic:
        B = B_asymptotic

    return B
