"""
Generator for the the initial population of neutron stars

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""

import logging
import os

import hydra
import pandas as pd

import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population as ipop

log = logging.getLogger(__name__)


@hydra.main()
def generate_population(cfg) -> None:

    """ Generate an initial population.

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
    log.info("Randomizing population age...")
    age = NS_population_initial.age()

    # Generating initial positions.
    log.info("Generating initial positions...")
    (
        r_initial,
        phi_initial,
        x_initial,
        y_initial,
        z_initial,
    ) = NS_population_initial.position(t_age=age)

    # Generating initial velocities summing the proper velocities to the orbital
    # velocities.
    log.info("Generating initial proper velocities...")
    (vp_r, vp_phi, vp_z,) = NS_population_initial.proper_velocity()

    log.info("Computing orbital velocities...")
    v_orb = NS_population_initial.orbital_velocity(r_initial, z_initial)

    v_r_initial = vp_r
    v_phi_initial = vp_phi + v_orb
    v_z_initial = vp_z

    # Adding the coordinates to a data frame for export.
    log.info("Creating data frame for exporting...")
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

    log.info(
        "Output generated in {}/{}".format(
            os.getcwd(), "initial_population.txt"
        )
    )


if __name__ == "__main__":

    generate_population()
