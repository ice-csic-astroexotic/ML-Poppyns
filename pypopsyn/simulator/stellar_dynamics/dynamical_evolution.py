"""
    Dynamical evolution of the neutron stars in the galactic potential

    We solve the system of dynamical differential equations in cylindrical coordinates,
    using a galactocentric reference frame. Here we are using the scipy.integrate.odeint
    package which uses the method 'LSODA' (Adams/BDF method with automatic stiffness
    detection and switching) from the Fortran library ODEPACK.

    We improve performance with Numba, which allows just-in-time (JIT) compilation.

    Authors:

            Vanessa Graber (graber@ice.csic.es)
            Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Tuple

import numpy as np
from numba import float64, jit
from scipy.integrate import odeint

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
from pypopsyn.simulator.config_simulator import cfg

gm.initialize_galactic_model()


@jit(
    float64[:](
        float64,
        float64[:],
        gm.galactic_model._numba_type_.class_type.instance_type,
    )
)
def dynamical_eq_system(
    t: float, initial_cond: np.ndarray, galactic_model: gm.GalaxyModelBase
) -> np.ndarray:
    """
    System of dynamical equations to solve to determine the orbits of the neutron
    stars in the galactic potential. The differential equation are written in
    cylindrical galactocentric coordinates (r, phi, z).

    Args:
        initial_cond (np.ndarray): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).

        t (float): unused time variable, required for the integration below.

        galactic_model (gm.GalaxyModelBase): a galactic model to calculate
        the needed potential.

    Returns:
         (np.ndarray): array of 6 values of the first order and second order
         derivatives at each time step.

    """

    r = initial_cond[0]
    z = initial_cond[2]

    gradient_mw_pot = galactic_model.cylind_coord_gradient_mw_potential(r, z)

    # First derivatives.
    dr_dt = initial_cond[3]
    dphi_dt = initial_cond[4]
    dz_dt = initial_cond[5]

    # Second derivatives.
    d2r_dt2 = r * dphi_dt**2 - gradient_mw_pot[0]
    d2phi_dt2 = -2 * dr_dt * dphi_dt / r - gradient_mw_pot[1]
    d2z_dt2 = -gradient_mw_pot[2]

    derivatives = np.array(
        [dr_dt, dphi_dt, dz_dt, d2r_dt2, d2phi_dt2, d2z_dt2]
    )

    return derivatives


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

    for i in range(n):

        # Linear time grid in years over which the dynamical evolution is performed;
        # each star's position and velocity is evolved for a time equal to its age.
        time_grid = np.append(
            np.arange(0.0, t_age[i], cfg["dyn_time_step"]),
            t_age[i],
        )

        # Save the odeint output which is a two-dimensional array of
        # shape (len(time_grid), 6).
        # We set tfirst=True to unify the structure of the input ODEs in order to be able
        # to compare different scipy functions to solve the ODEs.
        evol_output = np.array(
            odeint(
                dynamical_eq_system,
                y0=initial_cond[i],
                t=time_grid,
                args=(gm.galactic_model,),
                tfirst=True,
            )
        )

        if cfg["save_dyn_evolution"]:

            v_r_evol = evol_output[:, 3] * const.KPC_TO_KM / const.YR_TO_S
            v_phi_evol = (
                evol_output[:, 0]
                * evol_output[:, 4]
                * const.KPC_TO_KM
                / const.YR_TO_S
            )
            v_z_evol = evol_output[:, 5] * const.KPC_TO_KM / const.YR_TO_S

            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution = {
                i: {
                    "t": time_grid.tolist(),
                    "r(t)": evol_output[:, 0].tolist(),
                    "phi(t)": evol_output[:, 1].tolist(),
                    "z(t)": evol_output[:, 2].tolist(),
                    "v_r(t)": v_r_evol.tolist(),
                    "v_phi(t)": v_phi_evol.tolist(),
                    "v_z(t)": v_z_evol.tolist(),
                }
            }
            # Update the dictionary containing the evolution information of all the neutron stars.
            evolution_dictionary = {**evolution_dictionary, **evolution}

        # Save the final position and velocity.
        # Note: We save directly the v_phi velocity component and not the angular velocity omega.
        r_final[i] = evol_output[-1, 0]
        phi_final[i] = evol_output[-1, 1]
        z_final[i] = evol_output[-1, 2]
        v_r_final[i] = evol_output[-1, 3]
        omega_final = evol_output[-1, 4]
        v_phi_final[i] = omega_final * r_final[i]
        v_z_final[i] = evol_output[-1, 5]

    final_population = np.array(
        [r_final, phi_final, z_final, v_r_final, v_phi_final, v_z_final]
    ).T

    return final_population, evolution_dictionary
