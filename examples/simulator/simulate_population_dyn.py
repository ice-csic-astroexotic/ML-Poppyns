"""
Dynamically evolving a population of neutron stars.

An initial neutron star population of uniformly distributed ages is generated
and the respective objects evolved dynamically in time according to their age.

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

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import orjson
import pandas as pd

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


def simulate_population(args) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and dynamically evolving it forward in time.

    Args:

        output_path (pathlib.Path): Output directory for the run.
        json_override_path (pathlib.Path): Path to JSON with parameter overrides.

    Returns:

        Nothing.

    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update path-dependent configurations prepending the specified output path.
    cfg["profile_log"] = str(
        pathlib.Path().joinpath(output_path, cfg["profile_log"])
    )
    cfg["profile_json"] = str(
        pathlib.Path().joinpath(output_path, cfg["profile_json"])
    )

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_dyn"] is None:
        cfg["seed_dyn"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_dyn"]))
    np.random.seed(cfg["seed_dyn"])

    # Update simulator configuration with the provided JSON override (if any).
    cfg_override = {}
    if args.parameter_override:
        json_override_path = pathlib.Path(args.parameter_override)
        with open(json_override_path) as f:
            cfg_override = json.load(f)
            configuration.update_configuration(cfg_override)

    # Dump updated configuration to output path.
    config_dump_path = pathlib.Path().joinpath(
        output_path, "configuration.json"
    )
    with open(config_dump_path, "w") as f:
        json.dump(cfg, f, indent=4, sort_keys=True)

    # Initialize components of the simulator that need it.
    gm.initialize_galactic_model()
    sm.initialize_spiral_model()

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):

        # ===================== INITIALIZE THE POPULATION ========================

        with timewith.TimeWith(
            "[InitialPopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
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
                z_initial,
            ) = NS_population_initial.position(t_age=age)

            # Generating initial velocities by summing the kick
            # velocities at birth and the orbital velocities.
            log.info("Generating initial kick velocities...")
            (
                vk_r,
                vk_phi,
                vk_z,
            ) = NS_population_initial.kick_velocity()

            log.info("Computing orbital velocities...")
            v_orb = NS_population_initial.orbital_velocity(
                r_initial, z_initial
            )

            log.info("Computing initial total velocities...")
            v_r_initial = vk_r
            v_phi_initial = vk_phi + v_orb
            omega_initial = v_phi_initial / r_initial
            v_z_initial = vk_z

            # Compute the magnitude of the initial velocity vector for each star.
            v_initial = (
                np.sqrt(
                    v_r_initial**2 + v_phi_initial**2 + v_z_initial**2
                )
                * const.KPC_TO_KM
                / const.YR_TO_S
            )

            timer.checkpoint("[Initial position and velocity]")

            # Compute the total initial energy of the system.
            total_energy_initial = gm.galactic_model.total_energy(
                v_initial, r_initial, z_initial
            )

            timer.checkpoint("[Initial energy]")

            # Compute the initial z-component of the total angular momentum of the system.
            L_z_initial = gm.galactic_model.total_angular_momentum_z(
                v_phi_initial * const.KPC_TO_KM / const.YR_TO_S, r_initial
            )

            timer.checkpoint("[Initial Angular momentum]")

        # ===================== DYNAMICAL EVOLUTION ========================

        with timewith.TimeWith(
            "[EvolvePopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Evolve the initial population.
            log.info("Evolving the initial population in time...")

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

            # Evolve positions and velocities of the neutron stars forward in time.
            log.info("Evolving the positions and velocities...")
            final_population, dyn_evol_dict = dyn.dynamical_evolution(
                initial_cond, age
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

            timer.checkpoint("[Dynamic evolution]")

            # Compute the magnitude of the initial velocity vector for each star.
            v_final = np.sqrt(
                v_r_final**2 + v_phi_final**2 + v_z_final**2
            )

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
                f"Percentage variation of total energy of the system: {delta_energy_percentage} %"
            )

            timer.checkpoint("[Final energy]")

            # Compute the final z-component of the total angular momentum of the system.
            L_z_final = gm.galactic_model.total_angular_momentum_z(
                v_phi_final, r_final
            )

            # Compute the percentage variation in total energy during the simulation
            # with respect to the initial total energy.
            delta_Lz_percentage = (
                (L_z_final - L_z_initial) / L_z_initial * 100.0
            )

            log.info(
                f"Percentage variation of z-component of total angular momentum of the system: {delta_Lz_percentage} %"
            )

            timer.checkpoint("[Final angular momentum]")

            # ===================== EXPORT OUTPUT ========================

            # Adding the evolution output to a data frame for export.
            log.info("Creating data frame for exporting...")

            # Generating two header lines and merging them using MultiIndex.
            parameters_final = [
                "age",
                "r",
                "phi",
                "z",
                "v_r",
                "v_phi",
                "v_z",
            ]
            units_final = [
                "[yr]",
                "[kpc]",
                "[rad]",
                "[kpc]",
                "[km/s]",
                "[km/s]",
                "[km/s]",
            ]
            header_final = pd.MultiIndex.from_arrays(
                [parameters_final, units_final]
            )

            df_final = pd.DataFrame(
                data=np.array(
                    [
                        age,
                        r_final,
                        phi_final,
                        z_final,
                        v_r_final,
                        v_phi_final,
                        v_z_final,
                    ]
                ).T,
                columns=header_final,
            )

            # Save the data frame as a compressed binary file.
            final_output_path = pathlib.Path().joinpath(
                output_path,
                "final_pop_dyn.csv",
            )
            df_final.to_csv(final_output_path)

            log.info(
                f"Output of the evolved population generated in {os.getcwd()}/{final_output_path}"
            )

            timer.checkpoint("[Export]")

    # Cleanup. Reset seed to empty value.
    cfg["seed_dyn"] = None


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--output_dir",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory where the run will be saved.",
    )

    args.add_argument(
        "--parameter_override",
        nargs="?",
        type=str,
        default=None,
        help="Path to JSON containing the parameter override values.",
    )

    args = args.parse_args()

    simulate_population(args)
