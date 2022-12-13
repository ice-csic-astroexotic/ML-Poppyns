"""
Dynamical evolution of the neutron stars in the galactic potential.

We solve the system of dynamical differential equations in cylindrical coordinates,
using a galactocentric reference frame with Julia (see `julia_solver.jl` for more details).
Here we are using the Julia package OrdinaryDiffEq which uses the method 'LSODA' (Adams/BDF method with
automatic stiffness detection and switching).


Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Borja Miñano (borja.minano@uib.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)

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
from julia import Main

from pypopsyn.simulator.configuration import cfg


def dynamical_evolution(
    initial_cond: np.ndarray, t_age: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Performing the dynamical evolution of the neutron star population for a given
    galactic potential, starting from a set of initial conditions.

    Args:
        initial_cond (np.ndarray): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).

        t_age (np.ndarray): array of neutron star ages in [yr].

    Returns:
        (np.ndarray, dict): Tuple consisting of a two-dimensional array of shape (NS_number, 6)
        defining the neutron stars' final positions r [kpc], phi [rad], z [kpc] and velocities
        in [kpc/yr] in cylindrical coordinates and a dictionary containing the time evolution of
        these quantities for each neutron star (if the option to save the time evolution is enabled).
    """

    # Save the number of simulated neutron stars, which is flexible depending on whether
    # they are simulated all at once or one by one.
    n = len(t_age)

    # Initialization of a dictionary that will contain the evolution in time of
    # positions and velocities.
    evolution_dictionary = {}

    # Initialize the arrays that will contain the final positions
    # and velocities of the neutron stars.
    r_final = np.zeros(n)
    phi_final = np.zeros(n)
    z_final = np.zeros(n)
    v_r_final = np.zeros(n)
    v_phi_final = np.zeros(n)
    v_z_final = np.zeros(n)

    # Setting names in the ´Main´ module to send Python values to Julia.
    Main.galactic_model_input = cfg["galactic_model"]

    # Importing the ´julia_solver.jl´ file with Julia code into our Main Julia.
    Main.include(
        "pypopsyn/simulator_julia/stellar_dynamics/dynamical_evolution_solver.jl"
    )

    # Setting names in the ´Main´ module to send Python values to Julia.
    Main.n = n
    Main.initial_cond = initial_cond
    Main.time_step = cfg["dyn_time_step"]
    Main.t_age = t_age
    Main.save_dyn_evolution = cfg["save_dyn_evolution"]
    Main.tolerance = cfg["ODE_solver_tol"]

    # Solving the ODEs with Julia.
    (
        r_final,
        phi_final,
        z_final,
        v_r_final,
        v_phi_final,
        v_z_final,
        evolution_dictionary,
    ) = Main.eval("solver_calls()")

    final_population = np.array(
        [r_final, phi_final, z_final, v_r_final, v_phi_final, v_z_final]
    ).T

    return final_population, evolution_dictionary
