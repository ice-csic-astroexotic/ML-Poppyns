"""
Generator for the final population of neutron stars.
An initial population is generated and then evolved in time

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""

import hydra
import numpy as np
import pandas as pd

import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.dynamical_evolution as dyn
import pypopsyn.simulator.initial_population as ipop


@hydra.main()
def generate_population(cfg) -> None:

    """ Generate an initial population starting from some initial conditions and evolve
        it in time.

    Args:

        cfg: configuration dictionary for the simulator.

    Returns:

        Nothing.

    """

    # Update simulator configuration with the provided parameters.
    configuration.update_configuration(cfg)

    # generating an initial neutron star population
    NS_population_initial = ipop.InitialNeutronStarPopulation()

    # Generating ages.
    age = NS_population_initial.age()

    # Generating initial positions.
    (
        r_initial,
        phi_initial,
        x_initial,
        y_initial,
        z_initial,
    ) = NS_population_initial.position(t_age=age)

    # Generating initial velocities summing the proper velocities to the orbital
    # velocities.
    (vp_r, vp_phi, vp_z,) = NS_population_initial.proper_velocity()

    v_orb = NS_population_initial.orbital_velocity(r_initial, z_initial)

    v_r_initial = vp_r
    v_phi_initial = vp_phi + v_orb
    omega_initial = v_phi_initial / r_initial
    v_z_initial = vp_z

    # Adding the coordinates to a data frame for export.
    df_initial = pd.DataFrame(
        {
            "age": age,
            "r_initial": r_initial,
            "phi_initial": phi_initial,
            "x_initial": x_initial,
            "y_initial": y_initial,
            "z_initial": z_initial,
            "v_r_initial": v_r_initial,
            "v_phi_initial": v_phi_initial,
            "v_z_initial": v_z_initial,
            "vp_r": vp_r,
            "vp_phi": vp_phi,
            "vp_z": vp_z,
            "v_orb": v_orb,
        }
    )

    df_initial.columns = pd.MultiIndex.from_tuples(
        zip(
            df_initial.columns,
            [
                "[yr]",
                "[kpc]",
                "[rad]",
                "[kpc]",
                "[kpc]",
                "[kpc]",
                "[kpc / yr]",
                "[kpc /yr]",
                "[kpc / yr]",
                "[kpc / yr]",
                "[kpc /yr]",
                "[kpc / yr]",
                "[kpc / yr]",
            ],
        )
    )

    df_initial.to_csv("initial_population.txt", index=False, header=True)

    # evolve the initial population

    NS_number = len(age)

    # define the initial conditions
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

    # evolve the positions and velocities of the neutron stars in time
    final_population = dyn.dynamical_evolution(
        NS_number, initial_cond, age, time_step=1.0e4
    )

    r_final = final_population[:, 0]
    phi_final = final_population[:, 1]
    x_final = final_population[:, 2]
    y_final = final_population[:, 3]
    z_final = final_population[:, 4]
    v_r_final = final_population[:, 5]
    v_phi_final = final_population[:, 6]
    v_z_final = final_population[:, 7]

    # adding the coordinates to a data frame for export
    df_final = pd.DataFrame(
        {
            "r_final": r_final,
            "phi_final": phi_final,
            "x_final": x_final,
            "y_final": y_final,
            "z_final": z_final,
            "v_r_final": v_r_final,
            "v_phi_final": v_phi_final,
            "v_z_final": v_z_final,
        }
    )

    df_final.columns = pd.MultiIndex.from_tuples(
        zip(
            df_final.columns,
            [
                "[kpc]",
                "[rad]",
                "[kpc]",
                "[kpc]",
                "[kpc]",
                "[kpc / yr]",
                "[kpc /yr]",
                "[kpc / yr]",
            ],
        )
    )

    df_final.to_csv("final_population.txt", index=False, header=True)


if __name__ == "__main__":

    generate_population()
