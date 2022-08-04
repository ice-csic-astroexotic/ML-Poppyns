"""
Simulating a final population of neutron stars.

This implementation relies on solving ODEs using julia.

An initial neutron star population of uniformly distributed ages is generated
and the respective objects evolved in time according to their age.
Neutron stars are created and evolved one by one until a predefined number of
simulated stars is reached.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Borja Miñano (borja.minano @ uib.es)

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
import pandas as pd
from julia import Main

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population as ipop
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


def simulate_population(
    output_path: pathlib.Path, json_override_path: pathlib.Path = None
) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and evolving it forward in time.

    Args:

        output_path (pathlib.Path): Output directory for the run.
        json_override_path (pathlib.Path): Path to JSON with parameter overrides.

    Returns:

        Nothing.

    """

    # Update simulator configuration with the provided JSON override (if any).
    cfg_override = {}
    if json_override_path:
        with open(json_override_path) as f:
            cfg_override = json.load(f)
            configuration.update_configuration(cfg_override)

    # Dump configuration override to output path.
    override_dump_path = pathlib.Path().joinpath(output_path, "override.json")
    with open(override_dump_path, "w") as f:
        json.dump(cfg_override, f, indent=4, sort_keys=True)

    # Sending Python values to Julia to initialize some of the components needed for the simulator.
    Main.galactic_model_input = cfg["galactic_model"]

    # Importing the ´galactic_model.jl´ file with Julia code into our Main Julia.
    Main.include("pypopsyn/simulator_julia/galactic_model.jl")
    sm.initialize_spiral_model()

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_dyn"] is None:
        cfg["seed_dyn"] = int(time.time())

    # Set NumPy random set globally.
    log.info("Seed: {}".format(cfg["seed_dyn"]))
    np.random.seed(cfg["seed_dyn"])

    # Initialize the parameter lists.
    age = []
    r_initial = []
    phi_initial = []
    z_initial = []
    v_kick_r = []
    v_kick_phi = []
    v_kick_z = []
    v_orbital = []
    v_r_initial = []
    v_phi_initial = []
    v_z_initial = []
    B_initial = []
    chi_initial = []
    P_initial = []

    r_final = []
    phi_final = []
    z_final = []
    v_r_final = []
    v_phi_final = []
    v_z_final = []
    B_final = []
    chi_final = []
    P_final = []

    dyn_evolution_dictionary = {}
    magrot_evolution_dictionary = {}

    NS_count = 0

    with timewith.TimeWith(
        "[TotalSimulation]",
        configuration.cfg["profile_log"],
        configuration.cfg["profile_json"],
        configuration.cfg["show_profiling"],
    ) as timer:

        while NS_count < cfg["NS_number"]:

            # ===================== INITIALIZE THE POPULATION ========================

            log.info(f"Simulating neutron star number {NS_count}...")

            log.info("Generating neutron star initial conditions...")

            # Generate an initial neutron star.
            NS_initial = ipop.InitialNeutronStarPopulation(NS_number=1)

            # Generating age.
            t_age = NS_initial.age()
            age.append(float(t_age))

            # Generating initial position.
            (
                r_i,
                phi_i,
                z_i,
            ) = NS_initial.position(t_age=t_age, spiral_model=sm.spiral_model)

            r_initial.append(float(r_i))
            phi_initial.append(float(phi_i))
            z_initial.append(float(z_i))

            # Generating initial velocity by summing the kick
            # velocity at birth and the orbital velocity.
            (
                vk_r,
                vk_phi,
                vk_z,
            ) = NS_initial.kick_velocity()

            v_kick_r.append(float(vk_r))
            v_kick_phi.append(float(vk_phi))
            v_kick_z.append(float(vk_z))

            v_orb = NS_initial.orbital_velocity(r_i, z_i)

            v_orbital.append(float(v_orb))

            v_r_i = vk_r
            v_phi_i = vk_phi + v_orb
            omega_i = v_phi_i / r_i
            v_z_i = vk_z

            v_r_initial.append(float(v_r_i))
            v_phi_initial.append(float(v_phi_i))
            v_z_initial.append(float(v_z_i))

            # Computing the initial field strengths, misalignment angles, and periods
            B_i = NS_initial.magnetic_field()
            B_initial.append(float(B_i))

            chi_i = NS_initial.misalignment_angle()
            chi_initial.append(float(chi_i))

            P_i = NS_initial.period()
            P_initial.append(float(P_i))

            # ===================== DYNAMICAL EVOLUTION ========================

            # Define the initial conditions for the dynamical evolution.
            initial_cond = np.array([r_i, phi_i, z_i, v_r_i, omega_i, v_z_i]).T

            log.info("Evolving neutron star in time...")

            # Evolve position and velocity of the neutron star forward in time.
            NS_final, NS_dyn_evol_dict = dyn.dynamical_evolution(
                initial_cond, t_age
            )

            if configuration.cfg["save_dyn_evolution"]:
                # Update the dictionary containing the evolution information.
                NS_dyn_evol_dict[NS_count] = NS_dyn_evol_dict.pop(0)
                dyn_evolution_dictionary = {
                    **dyn_evolution_dictionary,
                    **NS_dyn_evol_dict,
                }

            r_final.append(float(NS_final[:, 0]))
            phi_final.append(float(NS_final[:, 1]))
            z_final.append(float(NS_final[:, 2]))
            v_r_final.append(float(NS_final[:, 3]))
            v_phi_final.append(float(NS_final[:, 4]))
            v_z_final.append(float(NS_final[:, 5]))

            # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================

            # Determine the evolved magnetic field, misalignment angle and rotation period.
            (
                B_f,
                chi_f,
                P_f,
                NS_magrot_evol_dict,
            ) = mre.magneto_rotational_evolution(
                B_i,
                chi_i,
                P_i,
                t_age,
            )

            if configuration.cfg["save_magrot_evolution"]:
                # Update the dictionary containing the evolution information.
                NS_magrot_evol_dict[NS_count] = NS_magrot_evol_dict.pop(0)
                magrot_evolution_dictionary = {
                    **magrot_evolution_dictionary,
                    **NS_magrot_evol_dict,
                }

            B_final.append(float(B_f))
            chi_final.append(float(chi_f))
            P_final.append(float(P_f))

            NS_count += 1

        timer.checkpoint("[Initialize and evolve population]")

        # Cleanup. Reset seed to empty value.
        configuration.cfg["seed_dyn"] = None

        log.info("Converting units and coordinate frames...")

        # Convert lists into numpy arrays.
        r_initial = np.array(r_initial)
        phi_initial = np.array(phi_initial)
        z_initial = np.array(z_initial)
        v_r_initial = np.array(v_r_initial)
        v_phi_initial = np.array(v_phi_initial)
        v_z_initial = np.array(v_z_initial)
        P_initial = np.array(P_initial)
        chi_initial = np.array(chi_initial)
        B_initial = np.array(B_initial)

        r_final = np.array(r_final)
        phi_final = np.array(phi_final)
        z_final = np.array(z_final)
        v_r_final = np.array(v_r_final)
        v_phi_final = np.array(v_phi_final)
        v_z_final = np.array(v_z_final)
        P_final = np.array(P_final)
        chi_final = np.array(chi_final)
        B_final = np.array(B_final)

        # Convert from polar coordinates to cartesian coordinates.
        x_initial, y_initial = coco.polar_to_cartesian(r_initial, phi_initial)
        x_final, y_final = coco.polar_to_cartesian(r_final, phi_final)

        # Convert initial and final velocities from [kpc/yr] into [km/s].
        v_r_initial = v_r_initial * const.KPC_TO_KM / const.YR_TO_S
        v_phi_initial = v_phi_initial * const.KPC_TO_KM / const.YR_TO_S
        v_z_initial = v_z_initial * const.KPC_TO_KM / const.YR_TO_S

        v_r_final = v_r_final * const.KPC_TO_KM / const.YR_TO_S
        v_phi_final = v_phi_final * const.KPC_TO_KM / const.YR_TO_S
        v_z_final = v_z_final * const.KPC_TO_KM / const.YR_TO_S

        # Convert final velocity components from galactocentric cylindrical coordinates
        # to galactocentric cartesian coordinates.
        (
            v_x_final,
            v_y_final,
            v_z_final,
        ) = coco.speed_cylindrical_to_cartesian(
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
        ) = coco.galactocentric_to_icrs(
            x_final, y_final, z_final, v_x_final, v_y_final, v_z_final
        )

        timer.checkpoint("[Units and coordinate conversions]")

        # Determining the period derivatives.
        log.info("Computing initial and final period derivatives...")
        period_derivative_vect = np.vectorize(pdv.period_derivative)
        P_dot_initial = period_derivative_vect(
            B_initial, chi_initial, P_initial
        )
        P_dot_final = period_derivative_vect(B_final, chi_final, P_final)

        timer.checkpoint("[Period derivatives]")

        log.info("Checking for energy conservation...")

        # Compute the magnitude of the initial velocity vector for each star in [km/s].
        v_initial = np.sqrt(
            v_r_initial**2 + v_phi_initial**2 + v_z_initial**2
        )

        # Compute the total initial energy of the system.

        # Setting names in the ´Main´ module to send Python values to Julia.
        Main.v_initial = v_initial
        Main.r_initial = r_initial
        Main.z_initial = z_initial

        # Evaluating the total energy function in Julia.
        total_energy_initial = Main.eval(
            "total_energy(v_initial, r_initial, z_initial)"
        )

        # Compute the magnitude of the final velocity vector for each star.
        v_final = np.sqrt(v_r_final**2 + v_phi_final**2 + v_z_final**2)

        # Compute the total energy of the system after the dynamical evolution.
        # Setting names in the ´Main´ module to send Python values to Julia.

        Main.v_final = v_final
        Main.r_final = r_final
        Main.z_final = z_final

        # Evaluating the total energy function in Julia.
        total_energy_final = Main.eval(
            "total_energy(v_final, r_final, z_final)"
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

        timer.checkpoint("[Energy conservation]")

        # ===================== EXPORT OUTPUT ========================

        # Adding the initial condition parameters to a data frame for export.
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
            "B",
            "chi",
            "P",
            "P_dot",
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
            "[G]",
            "[rad]",
            "[s]",
            "[s/yr]",
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
                    v_kick_r,
                    v_kick_phi,
                    v_kick_z,
                    v_orbital,
                    B_initial,
                    chi_initial,
                    P_initial,
                    P_dot_initial,
                ]
            ).T,
            columns=header_initial,
        )

        # Save the data frame as compressed binary file.
        initial_output_path = pathlib.Path().joinpath(
            output_path, "initial_population.pkl.gz"
        )
        df_initial.to_pickle(initial_output_path, compression="gzip")

        log.info(
            f"Output of the initial population generated in {os.getcwd()}/{initial_output_path}"
        )

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
            "B",
            "chi",
            "P",
            "P_dot",
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
            "[G]",
            "[rad]",
            "[s]",
            "[s/yr]",
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
                    B_final,
                    chi_final,
                    P_final,
                    P_dot_final,
                ]
            ).T,
            columns=header_final,
        )

        # Save the data frame as a compressed binary file.
        final_output_path = pathlib.Path().joinpath(
            output_path, "final_population.pkl.gz"
        )
        df_final.to_pickle(final_output_path, compression="gzip")

        log.info(
            f"Output of the evolved population generated in {os.getcwd()}/{final_output_path}"
        )

        if configuration.cfg["save_dyn_evolution"]:
            # Save dictionary containing evolution information to output path in a .json file.
            dyn_evolution_dump_path = pathlib.Path().joinpath(
                output_path, "dyn_evolution.json"
            )
            with open(dyn_evolution_dump_path, "w") as f:
                json.dump(
                    dyn_evolution_dictionary, f, indent=4, sort_keys=True
                )

            log.info(
                f"Detailed dynamical evolution output saved in {os.getcwd()}/{output_path}"
            )

        if configuration.cfg["save_magrot_evolution"]:
            # Save dictionary containing evolution information to output path in a .json file.
            magrot_evolution_dump_path = pathlib.Path().joinpath(
                output_path, "magrot_evolution.json"
            )
            with open(magrot_evolution_dump_path, "w") as f:
                json.dump(
                    magrot_evolution_dictionary, f, indent=4, sort_keys=True
                )

            log.info(
                f"Detailed magneto-rotational evolution output saved in {os.getcwd()}/{output_path}"
            )

        timer.checkpoint("[Export]")


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

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update path-dependent configurations prepending the specified output path.
    configuration.cfg["profile_log"] = pathlib.Path().joinpath(
        output_path, configuration.cfg["profile_log"]
    )
    configuration.cfg["profile_json"] = pathlib.Path().joinpath(
        output_path, configuration.cfg["profile_json"]
    )

    simulate_population(output_path, args.parameter_override)
