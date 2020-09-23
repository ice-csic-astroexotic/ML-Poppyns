"""
Simulating a final population of neutron stars.

An initial neutron star population of uniformly distributed ages is generated
and the respective objects evolved in time according to their age.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)

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

import logging
import os
import sys

import hydra
import numpy as np
import pandas as pd

import pypopsyn.benchmark.timefunc as timefunc
import pypopsyn.benchmark.timewith as timewith
import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.constants as const
import pypopsyn.simulator.coordinate_conversions as coord
import pypopsyn.simulator.dynamical_evolution as dyn
import pypopsyn.simulator.galactic_model as gm
import pypopsyn.simulator.initial_population as ipop

log = logging.getLogger(__name__)


@hydra.main()
@timefunc.time_function(
    configuration.cfg["profile_log"], configuration.cfg["show_profiling"]
)
def generate_population(cfg) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and evolving it forward in time.

    Args:

        cfg: configuration dictionary for the simulator.

    Returns:

        Nothing.

    """

    # Update simulator configuration with the provided parameters.
    configuration.update_configuration(cfg)

    # Initialize components of the simulator that need it.
    gm.initialize_galactic_model()

    with timewith.TimeWith(
        "[InitialPopulation]",
        configuration.cfg["profile_log"],
        configuration.cfg["profile_json"],
        configuration.cfg["show_profiling"],
    ) as timer:

        # Generate an initial neutron star population.
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

        # Generating initial velocities by summing the kick
        # velocities at birth and the orbital velocities.
        log.info("Generating initial kick velocities...")
        (vk_r, vk_phi, vk_z,) = NS_population_initial.kick_velocity()

        log.info("Computing orbital velocities...")
        v_orb = NS_population_initial.orbital_velocity(r_initial, z_initial)

        log.info("Computing initial total velocities...")
        v_r_initial = vk_r
        v_phi_initial = vk_phi + v_orb
        omega_initial = v_phi_initial / r_initial
        v_z_initial = vk_z

        # Compute the magnitude of the initial velocity vector for each star.
        v_initial = (
            np.sqrt(v_r_initial ** 2 + v_phi_initial ** 2 + v_z_initial ** 2)
            * const.KPC_TO_KM
            / const.YR_TO_S
        )

        timer.checkpoint("[Initial]")

        # Compute the initial total initial energy of the system.
        total_energy_initial = gm.galactic_model.total_energy(
            v_initial, r_initial, z_initial
        )

        timer.checkpoint("[Energy]")

        # Adding the parameters to a data frame for export.
        log.info("Creating data frame for exporting...")

        # Generating two header lines and merging them using MultiIndex.
        parameters_initial = [
            "age",
            "x",
            "y",
            "z",
            "vk_r",
            "vk_phi",
            "vk_z",
            "v_orb",
        ]
        units_initial = [
            "[yr]",
            "[kpc]",
            "[kpc]",
            "[kpc]",
            "[kpc/yr]",
            "[kpc/yr]",
            "[kpc/yr]",
            "[kpc/yr]",
        ]
        header_initial = pd.MultiIndex.from_arrays(
            [parameters_initial, units_initial]
        )

        df_initial = pd.DataFrame(
            data=np.array(
                [
                    age,
                    x_initial,
                    y_initial,
                    z_initial,
                    vk_r,
                    vk_phi,
                    vk_z,
                    v_orb,
                ]
            ).T,
            columns=header_initial,
        )

        # Save the data frame as compressed binary file.
        df_initial.to_pickle("initial_population.pkl.gz", compression="gzip")

        timer.checkpoint("[Export]")

        log.info(
            "Output of the initial population generated in {}/{}".format(
                os.getcwd(), "initial_population.pkl.gz"
            )
        )

    ############################################################################

    with timewith.TimeWith(
        "[EvolvePopulation]",
        configuration.cfg["profile_log"],
        configuration.cfg["profile_json"],
        configuration.cfg["show_profiling"],
    ) as timer:

        # Evolve the initial population.
        log.info("Evolving the initial population in time...")
        NS_number = len(age)

        # Define the initial conditions.
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

        # Evolve positions and velocities of the neutron stars forward in time.
        log.info("Evolving the positions and velocities...")
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

        # Convert velocities from [kpc/yr] into [km/s].
        v_r_final = v_r_final * const.KPC_TO_KM / const.YR_TO_S
        v_phi_final = v_phi_final * const.KPC_TO_KM / const.YR_TO_S
        v_z_final = v_z_final * const.KPC_TO_KM / const.YR_TO_S

        # Convert velocity component from galactocentric cylindrical coordinates
        # to galactocentric cartesian coordinates.
        v_x_final, v_y_final, v_z_final = coord.speed_cylindrical_to_cartesian(
            v_r_final, v_phi_final, v_z_final, phi_final
        )

        # Convert galactocentric coordinates and velocities into ICRS frame.
        (
            ra_final,
            dec_final,
            sun_dist,
            v_ra_final,
            v_dec_final,
            v_ls,
        ) = coord.galactocentric_to_icrs(
            x_final, y_final, z_final, v_x_final, v_y_final, v_z_final
        )

        timer.checkpoint("[Evolution]")

        # Compute the magnitude of the initial velocity vector for each star.
        v_final = np.sqrt(v_r_final ** 2 + v_phi_final ** 2 + v_z_final ** 2)

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

        log.info(
            "Percentage variation of total energy of the system: {} %".format(
                delta_energy_percentage
            )
        )

        timer.checkpoint("[Energy]")

        # Adding the evolution output to a data frame for export.
        log.info("Creating data frame for exporting...")

        # Generating two header lines and merging them using MultiIndex.
        parameters_final = [
            "age",
            "x",
            "y",
            "z",
            "RA",
            "DEC",
            "d",
            "v_r",
            "v_phi",
            "v_z",
            "v_RA",
            "v_DEC",
            "v_ls",
        ]
        units_final = [
            "[yr]",
            "[kpc]",
            "[kpc]",
            "[kpc]",
            "[deg]",
            "[deg]",
            "[kpc]",
            "[km/s]",
            "[km/s]",
            "[km/s]",
            "[mas/yr]",
            "[mas/yr]",
            "[km/s]",
        ]
        header_final = pd.MultiIndex.from_arrays(
            [parameters_final, units_final]
        )

        df_final = pd.DataFrame(
            data=np.array(
                [
                    age,
                    x_final,
                    y_final,
                    z_final,
                    ra_final,
                    dec_final,
                    sun_dist,
                    v_r_final,
                    v_phi_final,
                    v_z_final,
                    v_ra_final,
                    v_dec_final,
                    v_ls,
                ]
            ).T,
            columns=header_final,
        )

        # Save the data frame as a compressed binary file.
        df_final.to_pickle("final_population.pkl.gz", compression="gzip")

        log.info(
            "Output of the evolved population generated in {}/{}".format(
                os.getcwd(), "final_population.pkl.gz"
            )
        )

        timer.checkpoint("[Export]")

    # Cleanup. Reset seed to empty value.
    configuration.cfg["seed"] = None


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_population()
