"""
Dynamical evolution of the neutron stars in the galactic potential

We solve the system of dynamical differential equations in cylindrical coordinates,
using a galactocentric reference frame. Here we are using the scipy.integrate.odeint
package which uses the method 'LSODA' (Adams/BDF method with automatic stiffness
detection and switching) from the Fortran library ODEPACK.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

import numpy as np
from numba import jit
from scipy.integrate import odeint

import pypopsyn.simulator.coordinate_conversions as coco
import pypopsyn.simulator.galactic_model as gm


@jit
def dynamical_eq_system(initial_cond: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    System of dynamical equations to solve to determine the orbits of the neutron
    stars in the galactic potential. The differential equation are written in
    cylindrical galactocentric coordinates (r, phi, z).

    Args:
        initial_cond (np.ndarray): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0,
        v_z0) with
        the following units (kpc, rad, kpc, kpc/yr, rad/yr, kpc/yr)

        t (np.ndarray): time array in yr on which perform the integration

    Returns:
         (np.ndarray): array of 6 values of the first order and second order
         derivatives at each time step

    """

    r = initial_cond[0]
    z = initial_cond[2]

    gradient_mw_pot = gm.cylind_coord_gradient_mw_potential(r, z)

    # first derivatives
    dr_dt = initial_cond[3]
    dphi_dt = initial_cond[4]
    dz_dt = initial_cond[5]

    # second derivatives
    d2r_dt2 = r * dphi_dt ** 2 - gradient_mw_pot[0]
    d2phi_dt2 = -2 * dr_dt * dphi_dt / r
    d2z_dt2 = -gradient_mw_pot[2]

    derivatives = np.array(
        [dr_dt, dphi_dt, dz_dt, d2r_dt2, d2phi_dt2, d2z_dt2]
    )

    return derivatives


def dynamical_evolution(
    NS_number: int,
    initial_cond: np.ndarray,
    t_age: np.ndarray,
    time_step: (float) = 1.0e3,
) -> np.ndarray:
    """
    Perform the dynamical evolution in the galactic gravitational potential of the
    population of neutron stars, starting from some initial conditions.

    Args:
        NS_number (int): number of simulated neutron stars

        initial_cond (np.ndarray): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0) with
        the following units (kpc, rad, kpc, kpc/yr, rad/yr, kpc/yr)

        t_age (np.ndarray): array of ages in year of the neutron stars

        time_step (float): time step used for the integration of the system of ODE of motion

    Returns:
        (np.ndarray): two dimensional array of shape (NS_number, 8) defining
        the final position in Cartesian and cylindrical coordinates and velocities in cylindrical coordinates of the neutron stars
    """

    # initialize the arrays that will contain the final positions and velocities of
    # the neutron stars
    r_final = np.zeros(NS_number)
    phi_final = np.zeros(NS_number)
    x_final = np.zeros(NS_number)
    y_final = np.zeros(NS_number)
    z_final = np.zeros(NS_number)
    v_r_final = np.zeros(NS_number)
    v_phi_final = np.zeros(NS_number)
    v_z_final = np.zeros(NS_number)

    for i in range(NS_number):

        # linear time grid in years over which perform the dynamical evolution
        # each neutron star position and velocity is evolved for a time equal to its age
        time_grid = np.arange(0.0, t_age[i] + time_step, time_step)

        # save the odeint output which is a two-dimensional array of shape (
        # len(time_grid), 6)
        evol_output = np.array(
            odeint(dynamical_eq_system, initial_cond[i], time_grid)
        )

        # save the final position and velocity
        r_final[i] = evol_output[-1, 0]
        phi_final[i] = evol_output[-1, 1]
        z_final[i] = evol_output[-1, 2]
        v_r_final[i] = evol_output[-1, 3]
        omega_final = evol_output[-1, 4]
        v_phi_final[i] = omega_final * r_final[i]
        v_z_final[i] = evol_output[-1, 5]

        x_final[i], y_final[i] = coco.polar_to_cartesian(
            r_final[i], phi_final[i]
        )

    final_population = np.array(
        [
            r_final,
            phi_final,
            x_final,
            y_final,
            z_final,
            v_r_final,
            v_phi_final,
            v_z_final,
        ]
    ).T

    return final_population
