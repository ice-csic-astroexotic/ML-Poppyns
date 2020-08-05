"""
Evolution of the pulsar dipolar magnetic field.

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

import numpy as np
from scipy.integrate import solve_ivp

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.configuration import cfg


def timescale_ohmic(L: float, sigma: float) -> float:
    """
    Calculating the ohmic diffusion timescale for a given conductivity and characteristic magnetic
    field length scale. Note that for our purposes, we neglect the fact that both quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the ohmic diffusion process. For our choices see the configuration file.

    Args:
        L (float): characteristic length scale on which the magnetic field varies, measured in [cm].
        sigma (float): conductivity of the dominating dissipative process, measure in [1/s].

    Returns:
        (float): ohmic diffusion timescale in [yr].
    """

    tau_ohm = 4 * np.pi * sigma * L ** 2 / const.c ** 2 / const.YR_TO_S

    return tau_ohm


def timescale_Hall(B: float, L: float, n_e: float) -> float:
    """
    Calculating the Hall timescale for a given field strength, characteristic magnetic field length
    scale and electron density. Note that for our purposes, we neglect the fact that all quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the conservative Hall process. For our choices of L and n_e see the configuration file.
    B will be identified with the initial dipolar magnetic field components at the pulsars' pole.

    Args:
        B (float): (local) magnetic field strength, measured in [G].
        L (float): characteristic length scale on which the magnetic field varies, measured in [cm].
        n_e (float): electron density, measured in [g/cm^3].

    Returns:
        (float): Hall timescale in [yr].
    """

    tau_Hall = (
        4 * np.pi * const.e * n_e * L ** 2 / (const.c * B) / const.YR_TO_S
    )

    return tau_Hall


def field_derivative(t: float, B: float, B_initial: float) -> float:
    """
    Calculating the change in the magnetic field strength of a pulsar based on a simplified
    differential equation (see eq. (18) of Aguilera et al. (2008)) that captures the
    characteristics of more complication numerical simulations of pulsar magnetic field
    evolution, i.e., at early timescales the Hall evolution dominates, while at late times
    the exponential magnetic field decay due to Ohmic dissipation kicks in. Note that as
    explained in Aguilera et al. (2008) the Hall timescale corresponds to that of the initial
    field strength.

    Args:
        t (float): unused current time parameter in [yr], only needed for integration below.
        B (float): pulsar's magnetic field magnitudes evolving with time, measured in [G].
        B_initial (float): pulsar's initial magnetic field magnitudes, measured in [G].

    Returns:
        (np.ndarray): magnetic field derivatives for a simulated pulsars in [G/yr].
    """

    tau_ohm = timescale_ohmic(cfg["L"], cfg["sigma"])
    tau_Hall = timescale_Hall(B_initial, cfg["L"], cfg["n_e"])

    B_deriv = -B / tau_ohm - B ** 2 / (tau_Hall * B_initial)

    return B_deriv


def field_evolution(
    B_initial: np.ndarray, NS_number: int, t_age: np.ndarray, time_step: float
) -> np.ndarray:
    """
    Evolving the neutron stars' magnetic fields according to their respective ages
    forward in time to obtain their current magnetic field strengths.

    Args:
        B_initial (np.ndarray): pulsars' initial magnetic field magnitudes, measured in [G].
        NS_number (int): number of simulated neutron stars.
        t_age (np.ndarray): array of neutron star ages in [yr].
        time_step (float): time step used to integrate the differential equation, in [yr].

    Returns:
        (np.ndarray): current pulsar magnetic field strengths, in [G].
    """

    # Initialization of the array for the magnetic field strengths at the current time.
    B_final = np.zeros(NS_number)

    for i in range(NS_number):

        # Generating a time grid at which the solution is evaluated. We start to
        # evolve each star at its birth, corresponding to time 0, and do so for
        # its full age in steps of the specified time_step. To obtain the magnetic
        # field at the current time, we append the current age value.
        time_grid = np.append(np.arange(0, t_age[i], time_step), t_age[i])

        # To integrate the problem, we use scipy's solve_ivp function.
        # Note that the args parameter only works in scipy version >1.4.0.
        evol_output = solve_ivp(
            field_derivative,
            t_span=[0, t_age[i]],
            y0=np.array([B_initial[i]]),
            method="RK45",
            t_eval=time_grid,
            args=(B_initial[i],),
        )

        # The solution evaluated at the times t_eval=time_grid can be accessed via .y.
        # The last value in the array corresponds to the current field strength.
        B_final[i] = evol_output.y[0][-1]

    return B_final
