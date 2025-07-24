"""
    Dynamically evolving a population of neutron stars.

    An initial neutron star population of uniformly distributed ages is generated
    and the respective objects evolved dynamically in time according to their age.

    Display help message to run the code:

    python simulate_population_dyn.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np

import pypopsyn.simulator.config_simulator as configuration
import pypopsyn.simulator.initial_population as ipop
import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
import utilities.benchmark.timewith as timewith
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)


def simulate_population(args: argparse.Namespace) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and dynamically evolving it forward in time.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - save_dir (pathlib.Path): Output directory for the run.
            - json_override_path (pathlib.Path): Path to JSON with parameter overrides.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update path-dependent configurations prepending the specified output path.
    prof_log_path = pathlib.Path().joinpath(output_path, cfg["profile_log"])
    prof_json_path = pathlib.Path().joinpath(output_path, cfg["profile_json"])

    # Remove the profile.json and profile.log files to prevent interrupted server connections issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)

    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

    cfg["profile_json"] = str(prof_json_path)
    cfg["profile_log"] = str(prof_log_path)

    # Update simulator configuration with the provided JSON override (if any).
    if args.parameter_override:
        json_override_path = pathlib.Path(args.parameter_override)
        with open(json_override_path) as f:
            cfg_override = json.load(f)
            configuration.update_configuration(cfg_override)

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_dyn"] is None:
        cfg["seed_dyn"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_dyn"]))
    np.random.seed(cfg["seed_dyn"])

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):

        # ===================== INITIALIZE THE POPULATION ========================

        with timewith.TimeWith(
            "[InitializePopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            log.info("Initialize the dynamical properties...")

            # Initialize components of the simulator that need it.
            gm.initialize_galactic_model()
            if not cfg["sample_edm"]:
                sm.initialize_spiral_model()

            # Generate an initial neutron star population.
            NS_population_initial = ipop.InitialNeutronStarPopulation(
                cfg["NS_number"]
            )
            # Generating ages.
            age = NS_population_initial.age()

            # Initialize neutron star dynamical properties.
            pop_dyn_initial = dyn.initialize_population_dyn(age)

        # ===================== DYNAMICAL EVOLUTION ========================

        with timewith.TimeWith(
            "[DynamicalEvolution]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Evolve the initial population.
            log.info("Evolving the initial population in time...")

            pop_dyn_final = dyn.evolve_population_dyn(
                pop_dyn_initial, output_path
            )

            # Check the conservation of the z-component of the angular momentum and total energy:
            dyn.check_angular_momentum_energy_conservation(
                pop_dyn_initial, pop_dyn_final, log
            )

        # ===================== EXPORT OUTPUT ========================

        with timewith.TimeWith(
            "[Export]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):
            log.info("Creating data frame for exporting...")

            df_final = dyn.create_output_dataframe_dyn(pop_dyn_final)

            # Save the data frame as a compressed binary file.
            final_output_path = pathlib.Path().joinpath(
                output_path,
                "final_pop_dyn.csv",
            )
            df_final.to_csv(final_output_path)

            log.info(
                f"Output of the evolved population generated in {os.getcwd()}/{final_output_path}"
            )

    # Dump updated configuration to output path.
    config_dump_path = pathlib.Path().joinpath(
        output_path, "configuration.json"
    )
    with open(config_dump_path, "w") as f:
        json.dump(cfg, f, indent=4, sort_keys=True)

    # Reset seed, profile_log, and profile_json to default values. This is done to prevent issues when
    # calling the simulate_population function in other scripts more than once, ensuring that the values are
    # properly reset.

    cfg["seed_dyn"] = None
    cfg["profile_log"] = "profile.log"
    cfg["profile_json"] = "profile.json"


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="output/sim_dyn",
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
