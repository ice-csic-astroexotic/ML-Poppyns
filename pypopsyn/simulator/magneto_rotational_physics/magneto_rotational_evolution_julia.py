"""
Magneto-rotational evolution of the neutron stars

We solve the system of differential equations governing the evolution of the spin period and inclination angle.
Here we are using the Julia OrdinaryDiffEq package which uses the method
'LSODA' (Adams/BDF method with automatic stiffness
detection and switching) from the Fortran library ODEPACK.

Authors:

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

import time
from typing import Tuple

import numpy as np
from numba import cfunc, float64, jit, njit
from scipy.integrate import odeint, solve_ivp

import pypopsyn.simulator.basics.constants as const
from julia import Main
from pypopsyn.simulator.configuration import cfg


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

    # Initial conditions for the two parameters.
    initial_cond = np.column_stack((chi_initial, P_initial))

    # Initialization of a dictionary that will contain the evolution in time of
    # positions and velocities.
    evolution_dictionary = {}

    # Initialization of the array for the three parameters.
    B_final = np.zeros(n)
    chi_final = np.zeros(n)
    P_final = np.zeros(n)

    # Importing the ´magneto_rotational_evolution_solver.jl´ file with Julia code into our Main Julia.
    Main.include("julia/magneto_rotational_evolution_solver.jl")

    # Setting names in the ´Main´ module to send Python values to Julia.
    Main.NS_mass = cfg["NS_mass"]
    Main.NS_radius = cfg["NS_radius"]
    Main.k_coefficients_0 = cfg["k_coefficients"][0]
    Main.k_coefficients_1 = cfg["k_coefficients"][1]
    Main.k_coefficients_2 = cfg["k_coefficients"][2]

    Main.a1_cfg = cfg["a1"]
    Main.a2_cfg = cfg["a2"]
    Main.A1_cfg = cfg["A1"]
    Main.A2_cfg = cfg["A2"]
    Main.b1_cfg = cfg["b1"]
    Main.b2_cfg = cfg["b2"]
    Main.t_trans_cfg = cfg["t_trans"]
    Main.a_late_t_cfg = cfg["a_late_t"]
    Main.B_millisec_mean = cfg["B_millisec_mean"]
    Main.B_millisec_sigma = cfg["B_millisec_sigma"]

    Main.n = n
    Main.initial_cond_magrot = initial_cond
    Main.magrot_time_step_log = cfg["magrot_time_step_log10"]
    Main.t_age = t_age
    Main.save_magrot_evolution = cfg["save_magrot_evolution"]
    Main.tolerance = cfg["ODE_solver_tol"]

    # Solving the ODEs with Julia.
    (
        B_final,
        chi_final,
        P_final,
        evolution_dictionary,
    ) = Main.eval("solver_calls()")

    return B_final, chi_final, P_final, evolution_dictionary
