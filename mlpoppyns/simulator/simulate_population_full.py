"""
    Simulating a final population of neutron stars.

    An initial neutron star population of uniformly distributed ages is generated
    and the respective objects evolved in time according to their age.
    We simulate both the dynamical evolution in the Galaxy and the magneto-rotational
    evolution.
    Finally, we model the radio and X-ray emission and simulate the detection from radio and X-ray surveys.

    Display help message to run the code:

    python simulate_population_full.py --help

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

import mlpoppyns.simulator.config_simulator as configuration
import mlpoppyns.simulator.initial_population as ipop
import mlpoppyns.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
import mlpoppyns.simulator.multiband_emission.emission_radio as er
import mlpoppyns.simulator.multiband_emission.emission_xray as ex
import mlpoppyns.simulator.multiband_surveys.surveys_wrapper as sw
import mlpoppyns.simulator.stellar_dynamics.coordinate_conversions as coco
import mlpoppyns.simulator.stellar_dynamics.dynamical_evolution as dyn
import mlpoppyns.simulator.stellar_dynamics.galactic_model as gm
import mlpoppyns.simulator.stellar_dynamics.spiral_model as sm
import utilities.benchmark.timewith as timewith
import utilities.dataframe_builder as dfb
from mlpoppyns.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


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
    if cfg["seed_full"] is None:
        cfg["seed_full"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_full"]))
    np.random.seed(cfg["seed_full"])

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):

        # Initialize the surveys.
        SurveyData = sw.initialize_all_surveys()

        surveys_cfg = SurveyData.surveys_cfg
        surveys_radio = SurveyData.surveys_radio
        surveys_xray = SurveyData.surveys_xray

        # ===================== INITIALIZE THE POPULATION ========================

        with timewith.TimeWith(
            "[InitializePopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Generate an array of indices.
            NS_idx = np.arange(cfg["NS_number"], dtype=int)

            log.info(
                "Initialize the dynamical and magneto-rotational properties..."
            )

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

            # Initialize neutron star magneto-rotational properties.
            pop_magrot_initial = mre.initialize_population_magrot(age)

            pop_full_initial = pop_dyn_initial | pop_magrot_initial
            pop_full_initial["idx"] = NS_idx

            # Adding the parameters to a data frame for export.
            log.info("Creating data frame for exporting...")

            df_initial = dfb.create_output_dataframe_initial_pop(
                pop_full_initial
            )

            # Save the data frame as compressed binary file.
            initial_output_path = pathlib.Path().joinpath(
                output_path, "initial_population.pkl.gz"
            )
            df_initial.to_pickle(initial_output_path, compression="gzip")

            log.info(
                f"Output of the initial population generated in {os.getcwd()}/{initial_output_path}"
            )

        # ===================== DYNAMICAL EVOLUTION ========================

        with timewith.TimeWith(
            "[DynamicalEvolution]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Evolve the initial population.
            log.info("Evolve the initial population in time dynamically...")

            pop_dyn_final = dyn.evolve_population_dyn(
                pop_dyn_initial, output_path
            )

            # Check the conservation of the z-component of the angular momentum and total energy:
            dyn.check_angular_momentum_energy_conservation(
                pop_dyn_initial, pop_dyn_final, log
            )

            # Add the positions in all coordinates to the final dictionary containing the dynamical information.
            pop_dyn_final = coco.convert_cylindrical_to_all_coordinates(
                pop_dyn_final
            )

        # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================

        with timewith.TimeWith(
            "[MagnetoRotationalEvolution]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Evolve the magneto-rotational properties in time.
            log.info(
                "Evolving magnetic field, misalignment angle and rotation period..."
            )

            pop_magrot_final = mre.evolve_population_magrot(
                pop_magrot_initial, output_path
            )

            # Merge the two dictionaries containing the evolved dynamical and magneto-rotational properties.
            pop_final = pop_dyn_final | pop_magrot_final
            pop_final["idx"] = NS_idx

        # ===================== RADIO AND X-RAY EMISSION ========================

        with timewith.TimeWith(
            "[RadioXrayEmission]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Compute a dictionary containing the sky coverage masks for all the surveys.
            coverage_dict = sw.compute_surveys_coverage(
                surveys_radio,
                surveys_xray,
                pop_final,
                dist_cutoff=35.0,
            )

            # Add the survey sky coverage information to the final population dictionary.
            pop_final |= coverage_dict

            # Compute the properties of pulsars whose radio beam intercepts our line of sight.
            pop_radio = er.radio_population_intercepted(
                pop_final,
                pop_final["coverage_radio"],
                full_population=True,
            )

            # Determine fraction of pulsars beamed towards us.
            fraction_intercepted = len(
                pop_radio["intercepted_radio"][pop_radio["intercepted_radio"]]
            ) / len(pop_radio["intercepted_radio"])

            log.info(
                f"Fraction of pulsars beaming towards us in radio: {fraction_intercepted}"
            )

            pop_xray = {}

            if cfg["simulation_xray"]:
                Lx_interpolator = ex.load_Lx_interpolator()
                crust_failure_rate_interpolator = (
                    ex.load_crust_failure_rate_interpolator()
                )

                # Compute X-ray emission of neutron stars.
                pop_xray = ex.xray_population(
                    pop_final,
                    pop_final["coverage_xray"],
                    L_x_interpolator=Lx_interpolator,
                    crust_failure_rate_interpolator=crust_failure_rate_interpolator,
                    full_population=True,
                )

            # Merge the dictionary containing the intrinsic radio and X-ray emission properties.
            pop_full_final = pop_radio | pop_xray

            # Adding the parameters to a data frame for export.
            log.info("Creating data frame for exporting...")

            df_final = dfb.create_output_dataframe_final_pop(
                pop_full_final, cfg["simulation_xray"]
            )

            # Save the data frame as compressed binary file.
            final_output_path = pathlib.Path().joinpath(
                output_path, "final_population.pkl.gz"
            )
            df_final.to_pickle(final_output_path, compression="gzip")

            log.info(
                f"Output of the final population generated in {os.getcwd()}/{final_output_path}"
            )

        # ===================== RADIO AND X-RAY DETECTION ========================

        with timewith.TimeWith(
            "[RadioXrayDetection]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            fraction_coverage = (
                np.count_nonzero(coverage_dict["coverage_radio"])
                / cfg["NS_number"]
            )
            log.info(
                f"Fraction of pulsars in the sky region covered by the radio surveys: {fraction_coverage}"
            )

            # Simulating the radio survey.
            log.info("Simulate detection with the radio surveys...")

            # Filter the population to include only pulsars detected by the radio surveys.
            pop_detected_radio = sw.radio_detection(
                surveys_radio,
                pop_radio,
                pop_radio["intercepted_radio"],
                log,
                full_population=True,
            )

            sw.update_survey_data(
                SurveyData,
                pop_detected_radio,
                "radio",
                cfg["NS_number"],
                [],
                log,
            )

            if cfg["simulation_xray"]:
                # Simulating the radio survey.
                log.info("Simulate detection with the X-ray surveys...")

                # Filter the population to include only pulsars detected by the X-ray surveys.
                pop_detected_x = sw.xray_detection(
                    surveys_xray,
                    pop_xray,
                )

                sw.update_survey_data(
                    SurveyData,
                    pop_detected_x,
                    "X-ray",
                    cfg["NS_number"],
                    [],
                    log,
                )

        # ===================== EXPORT OUTPUT ========================

        # Adding the final output to a data frame for export.
        log.info("Creating data frame for exporting...")

        # Create output dataframes for each survey.
        dfs = sw.create_output_dataframe_surveys(
            SurveyData.dictionary_detected_radio,
            SurveyData.dictionary_detected_xray,
        )

        # Save the data frame as a compressed binary file.
        for survey in surveys_cfg:
            output_path_survey = (
                output_path / f"survey_{survey}_results.pkl.gz"
            )
            dfs[survey].to_pickle(output_path_survey, compression="gzip")
            log.info(
                f"Output of the detected population with {survey} generated in {os.getcwd()}/{output_path_survey}"
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

    cfg["seed_full"] = None
    cfg["profile_log"] = "profile.log"
    cfg["profile_json"] = "profile.json"


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="output/sim_full",
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
