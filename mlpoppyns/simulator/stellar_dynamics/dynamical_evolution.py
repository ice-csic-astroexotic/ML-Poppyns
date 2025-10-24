"""
    Dynamical evolution of the neutron stars in the galactic potential.

    We solve the system of dynamical differential equations in cylindrical coordinates,
    using a galactocentric reference frame. Here we are using the scipy.integrate.odeint
    package which uses the method 'LSODA' (Adams/BDF method with automatic stiffness
    detection and switching) from the Fortran library ODEPACK.

    We improve performance with Numba, which allows just-in-time (JIT) compilation.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import logging
import pathlib
from typing import Tuple

import numpy as np
import orjson
from numba import float64, jit
from scipy.integrate import odeint

import mlpoppyns.simulator.basics.constants as const
import mlpoppyns.simulator.initial_population as ipop
import mlpoppyns.simulator.stellar_dynamics.galactic_model as gm
from mlpoppyns.simulator.config_simulator import cfg

gm.initialize_galactic_model()


def initialize_population_dyn(age: np.ndarray) -> dict:
    """
    Initialize the dynamical properties of a neutron star population.

    Args:
        age (np.ndarray): An array of ages in [yr] for the neutron stars in the population to be initialized.

    Returns:
        (dict): A dictionary containing the initialized dynamical properties of the neutron star population.
    """

    # Initialize neutron star population properties.
    pop_initial = ipop.InitialNeutronStarPopulation(NS_number=len(age))

    # Generating initial positions.
    (
        r_initial,
        phi_initial,
        z_initial,
    ) = pop_initial.position(t_age=age)

    # Generating initial velocities by summing the kick
    # velocities at birth and the orbital velocities.
    (
        vk_r,
        vk_phi,
        vk_z,
    ) = pop_initial.kick_velocity()

    v_orb = pop_initial.orbital_velocity(r_initial, z_initial)

    v_r_initial = vk_r
    v_phi_initial = vk_phi + v_orb
    v_z_initial = vk_z

    dictionary_initial_pop_dyn = {
        "age": age,
        "r": r_initial,
        "phi": phi_initial,
        "z": z_initial,
        "v_r": v_r_initial,
        "v_phi": v_phi_initial,
        "v_z": v_z_initial,
        "v_orb": v_orb,
    }

    return dictionary_initial_pop_dyn


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
    System of dynamical equations to solve to determine the orbits of the neutron stars in the galactic potential.
    The differential equation are written in cylindrical galactocentric coordinates (r, phi, z).

    Args:
        t (float): Unused time variable, required for the integration below.
        initial_cond (np.ndarray): Array of 6 components defining the initial conditions in cylindrical coordinates
            (r0, phi0, z0, v_r0, omega0, v_z0) with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        galactic_model (gm.GalaxyModelBase): A galactic model to calculate the needed potential.

    Returns:
         (np.ndarray): Array of 6 values of the first order and second order derivatives at each time step.

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
        initial_cond (np.ndarray): Array of 6 components defining the initial
            conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
            with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        t_age (np.ndarray): Array of neutron star ages in [yr].

    Returns:
        (Tuple[np.ndarray, dict]): Tuple consisting of a two-dimensional array of shape (NS_number, 6)
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


def evolve_population_dyn(
    dict_pop_initial_dyn: dict,
    output_path: pathlib.Path,
) -> dict:
    """
    Evolve the dynamical properties of a neutron star population over time based on initial conditions.

    Args:
        dict_pop_initial_dyn (dict): Dictionary containing initial magneto-rotational properties of the population.
        output_path (pathlib.Path): The path where the evolution data will be saved if enabled in the configuration.

    Returns:
        (dict): A dictionary containing the properties of the evolved neutron star population.
    """
    age = dict_pop_initial_dyn["age"]
    r_initial = dict_pop_initial_dyn["r"]
    phi_initial = dict_pop_initial_dyn["phi"]
    z_initial = dict_pop_initial_dyn["z"]
    v_r_initial = dict_pop_initial_dyn["v_r"]
    v_phi_initial = dict_pop_initial_dyn["v_phi"]
    v_z_initial = dict_pop_initial_dyn["v_z"]

    omega_initial = v_phi_initial / r_initial

    # Define the initial conditions for the dynamical evolution.
    initial_cond = np.array(
        [
            r_initial,
            phi_initial,
            z_initial,
            v_r_initial,
            omega_initial,
            v_z_initial,
        ]
    ).T

    # Determine the evolved positions and velocities.
    final_population, dyn_evol_dict = dynamical_evolution(
        initial_cond,
        age,
    )

    r_final = final_population[:, 0]
    phi_final = final_population[:, 1]
    z_final = final_population[:, 2]
    v_r_final = final_population[:, 3]
    v_phi_final = final_population[:, 4]
    v_z_final = final_population[:, 5]

    # Convert velocities from [kpc/yr] into [km/s].
    v_r_final = v_r_final * const.KPC_TO_KM / const.YR_TO_S
    v_phi_final = v_phi_final * const.KPC_TO_KM / const.YR_TO_S
    v_z_final = v_z_final * const.KPC_TO_KM / const.YR_TO_S

    if cfg["save_dyn_evolution"]:
        # Save dictionary containing evolution information to output path in a .json file.
        dyn_evolution_dump_path = pathlib.Path().joinpath(
            output_path, "dyn_evolution.json"
        )

        with open(dyn_evolution_dump_path, "wb") as f:
            f.write(
                orjson.dumps(
                    dict(dyn_evol_dict),
                    option=orjson.OPT_SERIALIZE_NUMPY
                    | orjson.OPT_NON_STR_KEYS
                    | orjson.OPT_SORT_KEYS,
                )
            )

    dictionary_final_pop_dyn = {
        "age": age,
        "r": r_final,
        "phi": phi_final,
        "z": z_final,
        "v_r": v_r_final,
        "v_phi": v_phi_final,
        "v_z": v_z_final,
    }

    return dictionary_final_pop_dyn


def check_angular_momentum_energy_conservation(
    dict_pop_initial_dyn: dict,
    dict_pop_final_dyn: dict,
    logger: logging.Logger,
) -> None:
    """
    This function computes the total energy and angular momentum (L_z) of a
    stellar population before and after a dynamical evolution. It reports the
    percentage variation of each quantity to assess conservation.

    Args:
        dict_pop_initial_dyn (dict): Dictionary containing the initial dynamical
            properties of the population.
        dict_pop_final_dyn (dict): Dictionary containing the final dynamical
            properties of the population.
        logger (logging.Logger): Logger instance used to output conservation
            diagnostics (energy and angular momentum variations).
    """

    r_initial = dict_pop_initial_dyn["r"]
    z_initial = dict_pop_initial_dyn["z"]
    v_r_initial = dict_pop_initial_dyn["v_r"]
    v_phi_initial = dict_pop_initial_dyn["v_phi"]
    v_z_initial = dict_pop_initial_dyn["v_z"]

    # Compute the magnitude of the initial velocity vector for each star.
    v_initial = (
        np.sqrt(v_r_initial**2 + v_phi_initial**2 + v_z_initial**2)
        * const.KPC_TO_KM
        / const.YR_TO_S
    )

    # Compute the total initial energy of the system.
    total_energy_initial = gm.galactic_model.total_energy(
        v_initial, r_initial, z_initial
    )

    # Compute the initial z-component of the total angular momentum of the system.
    L_z_initial = gm.galactic_model.total_angular_momentum_z(
        v_phi_initial * const.KPC_TO_KM / const.YR_TO_S, r_initial
    )

    r_final = dict_pop_final_dyn["r"]
    z_final = dict_pop_final_dyn["z"]
    v_r_final = dict_pop_final_dyn["v_r"]
    v_phi_final = dict_pop_final_dyn["v_phi"]
    v_z_final = dict_pop_final_dyn["v_z"]

    # Compute the magnitude of the final velocity vector for each star.
    v_final = np.sqrt(v_r_final**2 + v_phi_final**2 + v_z_final**2)

    # Compute the total energy of the system after the dynamical evolution.
    total_energy_final = gm.galactic_model.total_energy(
        v_final, r_final, z_final
    )

    # Compute the percentage variation in total energy during the simulation
    # with respect to the initial total energy.
    delta_energy_percentage = (
        (total_energy_final - total_energy_initial)
        / total_energy_initial
        * 100.0
    )

    logger.info(
        f"Percentage variation of total energy of the system: {delta_energy_percentage} %"
    )

    # Compute the final z-component of the total angular momentum of the system.
    L_z_final = gm.galactic_model.total_angular_momentum_z(
        v_phi_final, r_final
    )

    # Compute the percentage variation in total energy during the simulation
    # with respect to the initial total energy.
    delta_Lz_percentage = (L_z_final - L_z_initial) / L_z_initial * 100.0

    logger.info(
        f"Percentage variation of z-component of total angular momentum of the system: {delta_Lz_percentage} %"
    )
