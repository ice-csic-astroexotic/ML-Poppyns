"""
    Simulating a detected population of neutron stars from a dynamically evolved population database.

    Display help message to run the code:

    python simulate_population_magrot_det.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
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
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_emission.emission_xray as ex
import pypopsyn.simulator.multiband_surveys.surveys_wrapper as sw
import pypopsyn.simulator.stellar_dynamics.load_dynamical_database as dyn
import utilities.benchmark.timewith as timewith
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def simulate_population(args) -> None:
    """
    Simulating a detected neutron star population starting from a dynamically evolved
    population database.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - dyn_data (str): Path to a dynamically evolved population database.
            - save_dir (str): Output directory for the run.
            - parameter_override (str): Path to JSON with parameter overrides.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update path-dependent configurations prepending the specified output path.
    prof_log_path = output_path / cfg["profile_log"]
    prof_json_path = output_path / cfg["profile_json"]

    # If already present, remove the profile.json and profile.log files to prevent
    # interrupted server connection issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)
    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

    # Update the paths to profile.json and profile.log files in the configuration file.
    cfg["profile_json"] = str(prof_json_path)
    cfg["profile_log"] = str(prof_log_path)

    # Update simulator configuration with the provided JSON override (if any).
    if args.parameter_override:
        json_override_path = pathlib.Path(args.parameter_override)
        with open(json_override_path) as f:
            cfg_override = json.load(f)
            configuration.update_configuration(cfg_override)

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_magrot"] is None:
        cfg["seed_magrot"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_magrot"]))
    np.random.seed(cfg["seed_magrot"])

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):
        SurveyData = sw.initialize_all_surveys()

        # Initialize the indicator for an excess in birth rate to False.
        cfg["birth_rate_excess"] = False

        # This variable progressively count how many stars we create in total.
        n_created = 0

        # List to store indices of detected neutron stars to avoid resampling.
        idx_remove = []

        surveys_cfg = SurveyData.surveys_cfg
        surveys_radio = SurveyData.surveys_radio
        surveys_xray = SurveyData.surveys_xray
        stop_flags = SurveyData.stop_flags

        # Loop to simulate stars until the detected number of pulsars for all surveys is reached.
        while not all(stop_flags.values()):

            # ===================== POPULATION INITIALIZATION ========================
            with timewith.TimeWith(
                "[InitializePopulation]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                n_batchsize = sw.adjust_n_batchsize(SurveyData)
                # Update the total number of simulated neutron stars.
                n_created += n_batchsize
                log.info(f"Total number of created neutron stars: {n_created}")

                # Load a batch of dynamically evolved neutron stars from the dynamical database.
                database_dyn_batch = dyn.load_database_dyn(
                    args.dyn_data, n_batchsize, idx_remove, log
                )

                # Compute the maximum simulation time in centuries.
                t_max = cfg["t_age_max"] / 100

                # Compute a dictionary containing the sky coverage masks for all the surveys.
                coverage_dict = sw.compute_surveys_coverage(
                    surveys_radio,
                    surveys_xray,
                    database_dyn_batch,
                    dist_cutoff=35.0,
                )

                # Filter the loaded database batch with the surveys' sky coverage.
                (
                    database_coverage,
                    idx_remove,
                ) = sw.apply_surveys_coverage_filter(
                    surveys_radio,
                    surveys_xray,
                    coverage_dict,
                    database_dyn_batch,
                    idx_remove,
                )

                # Initialize neutron star magneto-rotational properties.
                pop_magrot_initial = mre.initialize_population_magrot(
                    database_coverage["age"]
                )

                if cfg["simulation_xray"]:
                    # Initializing luminosity and crust failure rate interpolators once for use across all different
                    # detection loops in the simulation.
                    Lx_interpolator = ex.load_Lx_interpolator()
                    crust_failure_rate_interpolator = (
                        ex.load_crust_failure_rate_interpolator()
                    )

            # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================
            with timewith.TimeWith(
                "[MagnetoRotationalEvolution]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):
                log.info(
                    "Evolving magnetic field, misalignment angle and rotation period..."
                )

                # Evolve in time the magneto-rotational properties.
                pop_magrot_final = mre.evolve_population_magrot(
                    pop_magrot_initial, output_path
                )

                # Merge the dictionary containing the final magneto-rotational properties with the filtered dynamical
                # database.
                pop_final = database_coverage | pop_magrot_final

            # ===================== RADIO DETECTION ========================
            with timewith.TimeWith(
                "[RadioDetection]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                # Filter the population to include only pulsars whose radio beam intercepts our line of sight.
                pop_radio = er.radio_population_intercepted(
                    pop_final, pop_final["coverage_radio"]
                )
                if len(pop_radio["age"]) == 0:
                    break

                # Filter the population to include only pulsars detected by the radio surveys.
                pop_detected_radio_update = sw.radio_detection(
                    surveys_radio, pop_radio
                )

                sw.update_survey_data(
                    SurveyData,
                    pop_detected_radio_update,
                    "radio",
                    n_created,
                    idx_remove,
                    log,
                )

            # ===================== X DETECTION ========================
            if cfg["simulation_xray"]:
                with timewith.TimeWith(
                    "[XrayDetection]",
                    cfg["profile_log"],
                    cfg["profile_json"],
                    cfg["show_profiling"],
                ):
                    # Compute X-ray emission of neutron stars.
                    pop_xray = ex.xray_population(
                        pop_final,
                        pop_final["coverage_xray"],
                        L_x_interpolator=Lx_interpolator,
                        crust_failure_rate_interpolator=crust_failure_rate_interpolator,
                    )

                    # Filter the population to include only pulsars detected by the X-ray surveys.
                    pop_detected_x_update = sw.xray_detection(
                        surveys_xray,
                        pop_xray,
                    )

                    sw.update_survey_data(
                        SurveyData,
                        pop_detected_x_update,
                        "X-ray",
                        n_created,
                        idx_remove,
                        log,
                    )

            # ==========================================================

            # Compute the total current birth rate in NSs per century.
            birth_rate = n_created / t_max
            log.info(
                f"Galactic neutron star birth rate per century: {birth_rate} neutron stars per century."
            )
            # If the current birth rate exceeds the upper limit on the birth rate specified in the configuration file
            # stop the simulation.
            max_birth_rate = cfg["birth_rate_max"]
            if birth_rate > max_birth_rate:
                cfg["birth_rate_excess"] = True
                log.info(
                    f"Simulation stopped! Galactic neutron star birth rate exceeds {max_birth_rate} neutron stars per century."
                )
                break

        # Compute the neutron star birth rate for each survey.
        birth_rates = {}
        for survey in surveys_cfg:
            birth_rates[survey] = SurveyData.n_created_at_match[survey] / t_max

            log.info(
                f"Galactic neutron star birth rate per century according to {survey}: {birth_rates[survey]} neutron stars per century."
            )

            # Add the information of the birth rates and the number of detected neutron star to the configuration file.
            cfg[f"birth_rate_{survey}_at_match"] = birth_rates[survey]
            cfg[
                f"n_detected_sim_{survey}_at_match"
            ] = SurveyData.n_detected_sim_at_match[survey]
            cfg[f"n_detected_sim_{survey}_tot"] = SurveyData.n_detected_sim[
                survey
            ]

        # ===================== EXPORT OUTPUT ========================
        with timewith.TimeWith(
            "[Export]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):
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
            config_dump_path = pathlib.Path(output_path) / "configuration.json"
            with open(config_dump_path, "w") as f:
                json.dump(cfg, f, indent=4, sort_keys=True)

            # Reset seed, profile_log, and profile_json to default values. This is done to prevent issues when
            # calling the simulate_population function in other scripts more than once, ensuring that the values are
            # properly reset.

            cfg["seed_magrot"] = None
            cfg["profile_log"] = "profile.log"
            cfg["profile_json"] = "profile.json"


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--dyn_data",
        nargs="?",
        type=str,
        default="output/sim_dyn",
        help="Path to the file where the dynamically evolved population database is saved.",
    )

    args.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="data/example_simulation_dyn",
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
