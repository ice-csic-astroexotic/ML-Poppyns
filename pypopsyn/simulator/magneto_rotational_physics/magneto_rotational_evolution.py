"""
Combined evolution of the pulsar period, misalignment angle and magnetic field.

Authors:

        Vanessa Graber (graber@ice.csic.es)

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
from scipy.integrate import solve_ivp

import pypopsyn.simulator.magneto_rotational_physics.magnetic_field_derivative as mfdv
import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_derivative as madv
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv


def combined_derivatives(
    t: float, y: np.ndarray, B_initial: float
) -> np.ndarray:
    """
    Combining the three derivative functions for the magnetic field, the misalignment angle
    and the magnetic field (combined into a single three-component vector y) into a single
    function to allow combined integration.

    Args:
        t (float): unused time variable, required for the integration below.
        y (np.ndarray): three magneto-rotational parameters, i.e., B in [G], chi in [rad]
        and P in [s] for a single pulsar at a given time.
        B_initial (float): initial magnetic field magnitude for one pulsar, measured in [G].

    Returns:
        (np.ndarray): derivative of the three magneto-rotational parameters for one pulsar
    """

    # Unpacking the three components of the vector y.
    B, chi, P = y

    # Specifying the three derivatives.
    dy = np.zeros(len(y))

    dy[0] = mfdv.field_derivative(B, B_initial)
    dy[1] = madv.misalignment_angle_derivative(B, chi, P)
    dy[2] = pdv.period_derivative(B, chi, P)

    return dy


def magneto_rotational_evolution(
    B_initial: np.ndarray,
    chi_initial: np.ndarray,
    P_initial: np.ndarray,
    NS_number: int,
    t_age: np.ndarray,
    time_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        NS_number (int): number of simulated neutron stars.
        t_age (np.ndarray): array of neutron star ages in [yr].
        time_step (float): time step used to integrate the differential equation, in [yr].

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): current pulsar magnetic field strengths in [G],
        misalignment angles in [rad] and rotation periods in [s] for the simulated sample.
    """

    # Initialization of the array for the three parameters.
    B_final = np.zeros(NS_number)
    chi_final = np.zeros(NS_number)
    P_final = np.zeros(NS_number)

    # Initial conditions for the three parameters.
    y_initial = np.column_stack((B_initial, chi_initial, P_initial))

    for i in range(NS_number):

        # Generating a time grid at which the solution is evaluated. We start to
        # evolve each star at its birth, corresponding to time 0, and do so for
        # its full age in steps of the specified time_step. To obtain the magnetic
        # field at the current time, we append the current age value.
        time_grid = np.append(np.arange(0, t_age[i], time_step), t_age[i])

        # To integrate the problem, we use scipy's solve_ivp function.
        # Note that the args parameter only works in scipy version >1.4.0.
        evol_output = solve_ivp(
            combined_derivatives,
            t_span=[0, t_age[i]],
            y0=y_initial[i],
            method="RK45",
            t_eval=time_grid,
            args=(B_initial[i],),
        )

        # The solution evaluated at the times t_eval=time_grid can be accessed via .y.
        # The last value in the array corresponds to the current field strength.
        B_final[i] = evol_output.y[0][-1]
        chi_final[i] = evol_output.y[1][-1]
        P_final[i] = evol_output.y[2][-1]

    return B_final, chi_final, P_final
