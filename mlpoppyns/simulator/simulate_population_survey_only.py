"""
    Simulating a detected population of neutron stars.

    After loading a fullly evolved population of neutron stars in the Galaxy, we model the radio emission and
    simulate the detection from three radio surveys, namely the Parkes Multibeam Pulsar Survey (PMPS) and
    Swinburne Intermediate-latitude Pulsar Survey (SMPS) and the low- and mid-latitude High Time Resolution
    Universe (HTRU) surveys as well as an X-ray survey.

    Display help message to run the code:

    python simulate_population_survey_only.py --help

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
import shutil
import sys
import time

import numpy as np
import pandas as pd

import mlpoppyns.simulator.multiband_surveys.surveys_wrapper as sw
import utilities.benchmark.timewith as timewith
from mlpoppyns.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def simulate_surveys(args: argparse.Namespace) -> None:
    """
    Applying detection filters from radio and X-ray surveys to an evolved neutron star population.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - save_dir (pathlib.Path): Path to the directory where to save the output results.
            - full_data (pathlib.Path): Path to the compressed pickle file containing the full evolved population.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Check if the parsed final_population file exists. Otherwise, exit the script.
    full_path = pathlib.Path(args.full_data)
    final_population_path = full_path / "final_population.pkl.gz"

    if not final_population_path.exists():
        log.error(f"File {final_population_path} not found...")
        sys.exit()

    final_population_config_path = full_path / "configuration.json"

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update path-dependent configurations prepending the specified output path.
    prof_log_path = pathlib.Path().joinpath(output_path, cfg["profile_log"])
    prof_json_path = pathlib.Path().joinpath(output_path, cfg["profile_json"])

    # Remove the profile.json and profile.log files to prevent interrupted server connection issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)

    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

    cfg["profile_json"] = str(prof_json_path)
    cfg["profile_log"] = str(prof_log_path)

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

        # ===================== LOAD THE FULL FINAL POPULATION FILE ========================

        with timewith.TimeWith(
            "[LoadFinalPopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Load the final_population.pkl.gz file.
            final_population_location = pathlib.Path().joinpath(
                cfg["path_to_software"],
                final_population_path,
            )
            df_final_pop = pd.read_pickle(
                final_population_location,
                compression="gzip",
            )

            # Remove the second line of the header and convert the dataframe to a dictionary.
            df_final_pop.columns = df_final_pop.columns.get_level_values(0)
            pop_final = {
                col: df_final_pop[col].to_numpy()
                for col in df_final_pop.columns
            }

            pop_final["idx"] = df_final_pop.index.values

            # Load the various data columns of the final population file.
            log.info("Extracting indices...")
            NS_idx = df_final_pop.index.to_numpy()

            # Total number of neutron stars in the simulation.
            NS_number = len(NS_idx)

            # Compute a dictionary containing the sky coverage masks for all the surveys.
            coverage_dict = sw.compute_surveys_coverage(
                surveys_radio,
                surveys_xray,
                pop_final,
                dist_cutoff=35.0,
            )

            # Add the survey sky coverage information to the final population dictionary.
            pop_final |= coverage_dict

        # ===================== RADIO AND X-RAY DETECTION ========================

        with timewith.TimeWith(
            "[RadioXrayDetection]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Simulating the radio survey.
            log.info("Simulate detection with the radio surveys...")

            # Filter the population to include only pulsars detected by the radio surveys.
            pop_detected_radio = sw.radio_detection(
                surveys_radio,
                pop_final,
                pop_final["intercepted_radio"],
                log,
                full_population=True,
            )

            sw.update_survey_data(
                SurveyData,
                pop_detected_radio,
                "radio",
                NS_number,
                [],
                log,
            )

            if cfg["simulation_xray"]:
                # Simulating the X-ray survey.
                log.info("Simulate detection with the X-ray surveys...")

                # Filter the population to include only pulsars detected by the X-ray surveys.
                pop_detected_x = sw.xray_detection(
                    surveys_xray,
                    pop_final,
                )

                sw.update_survey_data(
                    SurveyData,
                    pop_detected_x,
                    "X-ray",
                    NS_number,
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

    # Copy the configuration file of the full simulation in the output folder for reproducibility purposes.
    config_dump_path = pathlib.Path(output_path) / "configuration.json"
    shutil.copy2(final_population_config_path, config_dump_path)

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
        "--full_data",
        nargs="?",
        type=str,
        default="output/sim_full",
        help="Path to the location where a final_population.pkl.gz file is saved.",
    )

    args = args.parse_args()

    simulate_surveys(args)
