"""
Combined evolution of the pulsar period, misalignment angle and magnetic field.

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

from typing import Tuple

import numpy as np
import scipy.interpolate
from numba import float64, jit, prange
from scipy import interpolate
from scipy.integrate import odeint

import pypopsyn.simulator.basics.interpolator as itp
import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_derivative as madv
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
from pypopsyn.simulator.configuration import cfg


def magnetic_field_evolution_fit_numpy(
    B_initial: float, t: np.ndarray, a: float, b: float, B_asymptotic: float
) -> np.ndarray:
    """
    An analytical function for the magnetic field evolution curves from the magneto-thermal evolution simulations.

    Args:
        B_initial(float): initial magnetic field strength in [G].
        t(np.ndarray): time in [s].
        a(float): power law index of the early time evolution.
        b(float): power law index of the late time evolution.
        B_asymptotic(float): asymptotic magnetic field strength at late times in [G].

    Returns:
        (np.ndarray): magnetic field evolution in [G] as a function of time t.
    """
    # Define the two timescales as a function of the initial B field.
    tau1 = (9.0e16 / B_initial) ** 1.0
    tau2 = (3.0e29 / B_initial) ** 0.4

    B = (B_initial - B_asymptotic) * (1 + t / tau1) ** (-a) * (
        1 + t / tau2
    ) ** (-b) + B_asymptotic

    return B


@jit([float64(float64, float64, float64, float64, float64)], nopython=True)
def magnetic_field_evolution_fit(
    B_initial: float, t: float, a: float, b: float, B_asymptotic: float
) -> float:
    """
    An analytical fit for the magnetic field evolution curves from the magneto-thermal evolution simulations.

    Args:
        B_initial(float): initial magnetic field strength in [G].
        t(float): time in [s].
        a(float): power law index of the early time evolution.
        b(float): power law index of the late time evolution.
        B_asymptotic(float): asymptotic magnetic field strength at late times in [G].

    Returns:
        (float): magnetic field value in [G] at time t.
    """
    # Define the two timescales as a function of the initial B field.
    tau1 = (9.0e16 / B_initial) ** 1.0
    tau2 = (3.0e29 / B_initial) ** 0.4

    B = (B_initial - B_asymptotic) * (1 + t / tau1) ** (-a) * (
        1 + t / tau2
    ) ** (-b) + B_asymptotic

    return B


@jit(
    [float64[:](float64, float64[:], float64, float64, float64, float64)],
    nopython=True,
)
def combined_derivatives(
    t: float,
    y: np.ndarray,
    B_initial: float,
    a: float,
    b: float,
    B_asymptotic: float,
) -> np.ndarray:
    """
    Combining the two derivative functions for the misalignment angle
    and the spin period (combined into a single two-component vector y) into a single
    function to allow combined integration.

    Args:
        t (float): unused time variable, required for the integration below.
        y (np.ndarray): two magneto-rotational parameters, i.e., chi in [rad]
        and P in [s] for a single pulsar at a given time.
        B_initial (float): initial magnetic field magnitude for one pulsar, measured in [G].
        a(float): power law index of the early time evolution.
        b(float): power law index of the late time evolution.
        B_asymptotic(float): asymptotic magnetic field strength at late times in [G].

    Returns:
        (np.ndarray): derivative of the two magneto-rotational parameters for one pulsar,
        quantities are referred to in respective changes per [yr].
    """

    # Unpacking the two components of the vector y.
    chi, P = y

    # Specifying the two derivatives.
    dy = np.zeros(len(y), dtype=np.float64)

    B = magnetic_field_evolution_fit(B_initial, t, a, b, B_asymptotic)

    dy[0] = madv.misalignment_angle_derivative(B, chi, P)
    dy[1] = pdv.period_derivative(B, chi, P)

    return dy


def magneto_rotational_evolution(
    B_initial: np.ndarray,
    chi_initial: np.ndarray,
    P_initial: np.ndarray,
    t_age: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Evolving the neutron stars' magnetic fields, misalignment angles and periods
    according to their respective ages forward in time to obtain their current
    magnetic field strengths, misalignment angles and periods. Note that right
    now the times at which these three parameters are evaluated (apart from the
    current time) do not agree for pulsars.

    Args:
        B_initial (np.ndarray): pulsars' initial magnetic field magnitudes, measured in [G].
        chi_initial (np.ndarray): pulsars' initial misalignment angles, measured in [rad].
        P_initial (np.ndarray): pulsars' initial rotation periods, measured in [s].
        t_age (np.ndarray): array of neutron star ages in [yr].

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, dict): Tuple consisting of three arrays
        defining the neutron stars' final magnetic field strengths in [G],
        misalignment angles in [rad] and rotation periods in [s] and a dictionary containing
        the time evolution of these quantities for each neutron star (if the option to save
        the time evolution is enabled).
    """

    # Save the number of simulated neutron stars, which is flexible depending if they are simulated
    # all at once or one by one.
    n = len(t_age)

    # Initialization of a dictionary that will contain the evolution in time of B, chi and P.
    evolution_dictionary = {}

    # Initialization of the array for the three parameters.
    B_final = np.zeros(n)
    chi_final = np.zeros(n)
    P_final = np.zeros(n)

    # Initial conditions for the three parameters.
    y_initial = np.column_stack((chi_initial, P_initial))

    # Draw a random asymptotic value of the magnetic field at late time from a log-Normal distribution.
    # This asymptotic value is based on the distribution of inferred magnetic fields for the old population
    # of millisecond pulsars.
    mean = 8.5
    sigma = 0.5
    B_asymptotic = 10 ** np.random.normal(mean, sigma, n)

    for i in range(n):

        # Generating a time grid at which the solution is evaluated. We start to
        # evolve each star at its birth, corresponding to time 0, and do so for
        # its full age in steps of the time_step specified in the configuration
        # file. To obtain the magnetic field at the current time, we append the
        # current age value.
        time_grid = np.append(
            10
            ** np.arange(
                0,
                np.log10(t_age[i]),
                cfg["magrot_time_step_log10"],
            ),
            t_age[i],
        )

        # To integrate the problem, we use scipy's odeint function.
        # We set tfirst=True to unify the structure of the input ODEs in order to be able
        # to compare different scipy functions to solve the ODEs.
        evol_output = np.array(
            odeint(
                combined_derivatives,
                y0=y_initial[i],
                t=time_grid,
                args=(B_initial[i], 0.14, 3.0, B_asymptotic[i]),
                tfirst=True,
            )
        )

        # Evaluate the magnetic field evolution.
        B_t = magnetic_field_evolution_fit_numpy(
            B_initial[i], time_grid, 0.14, 3.0, B_asymptotic[i]
        )

        if cfg["save_magrot_evolution"]:
            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution = {
                i: {
                    "t": time_grid.tolist(),
                    "B(t)": B_t.tolist(),
                    "chi(t)": evol_output[:, 0].tolist(),
                    "P(t)": evol_output[:, 1].tolist(),
                }
            }
            # Update the dictionary containing the evolution information.
            evolution_dictionary = {**evolution_dictionary, **evolution}

        # The solution evaluated at the times t_eval=time_grid can be accessed via .y.
        # The last value in the array corresponds to the current field strength.
        B_final[i] = B_t[-1]
        chi_final[i] = evol_output[-1, 0]
        P_final[i] = evol_output[-1, 1]

    return B_final, chi_final, P_final, evolution_dictionary
