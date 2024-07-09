"""
    Combined evolution of the pulsar period, misalignment angle and magnetic field.

    Authors:

            Vanessa Graber (graber@ice.csic.es)
            Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Tuple

import numpy as np
from numba import float64, jit
from scipy.integrate import odeint

import pypopsyn.simulator.magneto_rotational_physics.magnetic_field_derivative as mfdv
import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_derivative as madv
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
from pypopsyn.simulator.config_simulator import cfg


@jit(float64[:](float64, float64[:], float64))
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
        (np.ndarray): derivative of the three magneto-rotational parameters for one pulsar,
        quantities are referred to in respective changes per [yr].
    """

    # Unpacking the three components of the vector y.
    B, chi, P = y

    # Specifying the three derivatives.
    dy = np.zeros(len(y), dtype=np.float64)

    dy[0] = mfdv.field_derivative(B, B_initial)
    dy[1] = madv.misalignment_angle_derivative(B, chi, P)
    dy[2] = pdv.period_derivative(B, chi, P)

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
        (np.ndarray, np.ndarray, np.ndarray, dict): Tuple consisting of three arrays defining the neutron stars'
        final magnetic field strengths in [G], misalignment angles in [rad] and rotation periods in [s] and a
        dictionary containing the time evolution of these quantities for each neutron star (if the option to save
        the time evolution is enabled).
    """

    # Save the number of simulated neutron stars, which is flexible depending on whether
    # they are simulated all at once or one by one.
    n = len(t_age)

    # Initialization of a dictionary that will contain the evolution in time of B, chi and P.
    evolution_dictionary = {}

    # Initialization of the array for the three parameters.
    B_final = np.zeros(n)
    chi_final = np.zeros(n)
    P_final = np.zeros(n)

    # Initial conditions for the three parameters.
    y_initial = np.column_stack((B_initial, chi_initial, P_initial))

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
                args=(B_initial[i],),
                tfirst=True,
            )
        )

        if cfg["save_magrot_evolution"]:
            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution = {
                i: {
                    "t": time_grid.tolist(),
                    "B(t)": evol_output[:, 0].tolist(),
                    "chi(t)": evol_output[:, 1].tolist(),
                    "P(t)": evol_output[:, 2].tolist(),
                }
            }
            # Update the dictionary containing the evolution information.
            evolution_dictionary = {**evolution_dictionary, **evolution}

        # The solution evaluated at the times t_eval=time_grid can be accessed via .y.
        # The last value in the array corresponds to the current field strength.
        B_final[i] = evol_output[-1, 0]
        chi_final[i] = evol_output[-1, 1]
        P_final[i] = evol_output[-1, 2]

    return B_final, chi_final, P_final, evolution_dictionary
