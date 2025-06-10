"""
    Simulating a range of surveys based on a simulation run performed with the full simulation scripts.

    Display help message to run the code:

    python simulate_population_full.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Vanessa Graber (Vanessa.Graber@rhul.ac.uk)
"""

import argparse
import logging
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import utilities.benchmark.timewith as timewith
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def simulate_surveys(args: argparse.Namespace) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and dynamically evolving it forward in time.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - save_dir (pathlib.Path): Output directory for the run.
            - full_data (pathlib.Path): Directory where a final_population.pkl.gz file is saved.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # If the output directory does not exist, create it.
    output_path = pathlib.Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if the parsed final_population file exists. Otherwise, exit the script.
    full_path = pathlib.Path(args.full_data)
    final_population_path = full_path / "final_population.pkl.gz"

    if not final_population_path.exists():
        log.error(f"File {final_population_path} not found...")
        sys.exit()

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

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_full"] is None:
        cfg["seed_full"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_full"]))
    np.random.seed(cfg["seed_full"])

    # Initialize the surveys.
    PMPS_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
    )

    SMPS_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/Swinburne_Parkes_parameters.json",
    )

    HTRU_low_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
    )

    HTRU_mid_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
    )

    HTRU_high_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/htru_high_parameters.json",
    )

    SKA_low_AAstar_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_low_parameters_AAstar.json",
    )

    SKA_low_AA4_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_low_parameters_AA4.json",
    )

    SKA_mid_band1_AAstar_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_mid_parameters_band1_AAstar.json",
    )

    SKA_mid_band2_AAstar_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_mid_parameters_band2_AAstar.json",
    )

    SKA_mid_band1_AA4_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_mid_parameters_band1_AA4.json",
    )

    SKA_mid_band2_AA4_par_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/SKA_mid_parameters_band2_AA4.json",
    )

    survey_PMPS = sr.SurveyRadio(PMPS_par_path)
    survey_SMPS = sr.SurveyRadio(SMPS_par_path)
    survey_HTRU_low = sr.SurveyRadio(HTRU_low_par_path)
    survey_HTRU_mid = sr.SurveyRadio(HTRU_mid_par_path)
    survey_HTRU_high = sr.SurveyRadio(HTRU_high_par_path)
    survey_SKA_low_AAstar = sr.SurveyRadio(SKA_low_AAstar_par_path)
    survey_SKA_low_AA4 = sr.SurveyRadio(SKA_low_AA4_par_path)
    survey_SKA_mid_band1_AAstar = sr.SurveyRadio(SKA_mid_band1_AAstar_par_path)
    survey_SKA_mid_band2_AAstar = sr.SurveyRadio(SKA_mid_band2_AAstar_par_path)
    survey_SKA_mid_band1_AA4 = sr.SurveyRadio(SKA_mid_band1_AA4_par_path)
    survey_SKA_mid_band2_AA4 = sr.SurveyRadio(SKA_mid_band2_AA4_par_path)

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):

        # ===================== LOAD THE FULL FINAL POPULATION FILE ========================

        with timewith.TimeWith(
            "[LoadFinalPopulation]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Load the final_population.pkl.gz.
            final_population_location = pathlib.Path().joinpath(
                cfg["path_to_software"],
                final_population_path,
            )
            df_final_pop = pd.read_pickle(
                final_population_location,
                compression="gzip",
            )

            # Load the various data columns of the final population file.
            log.info("Extracting indices...")
            NS_idx = df_final_pop.index.to_numpy()

            # Total number of neutron stars in the simulation.
            NS_number = len(NS_idx)

            log.info("Extracting positional information...")
            RA_final = df_final_pop["RA"]["[deg]"].to_numpy()
            DEC_final = df_final_pop["DEC"]["[deg]"].to_numpy()
            l_final = df_final_pop["l"]["[deg]"].to_numpy()
            b_final = df_final_pop["b"]["[deg]"].to_numpy()
            d_final = df_final_pop["d"]["[kpc]"].to_numpy()

            log.info("Extracting velocity information...")
            pm_RA_final = df_final_pop["pm_RA"]["[mas yr^-1]"].to_numpy()
            pm_DEC_final = df_final_pop["pm_DEC"]["[mas yr^-1]"].to_numpy()

            log.info("Extracting magneto-rotational information...")
            P_final = df_final_pop["P"]["[s]"].to_numpy()
            P_dot_final = df_final_pop["P_dot"]["[s s^-1]"].to_numpy()

            log.info("Extracting luminosity information...")
            S_radio_bol_final = df_final_pop["S_radio_bol"][
                "[erg s^-1 cm^(-2)]"
            ].to_numpy()
            w_int_final = df_final_pop["w_int"]["[s]"].to_numpy()
            intercepted_radio_final = df_final_pop["intercepted_radio"][
                " "
            ].to_numpy()
            DM_final = df_final_pop["DM"]["[pc cm^-3]"].to_numpy()
            tau_sc_final = df_final_pop["tau_sc"]["[s]"].to_numpy()
            spectral_index_final = df_final_pop["spectral_index"][
                " "
            ].to_numpy()

            timer.checkpoint("[Data extracted]")

        # ===================== RADIO EMISSION ========================

        with timewith.TimeWith(
            "[RadioEmission]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Convert array with 0 and 1 to Boolean values.
            intercepted_radio_final = intercepted_radio_final.astype(bool)

            # Determine fraction of pulsars beamed towards us.
            fraction_intercepted = len(
                intercepted_radio_final[intercepted_radio_final]
            ) / len(intercepted_radio_final)

            log.info(
                f"Fraction of pulsars beaming towards us in radio: {fraction_intercepted}"
            )

            timer.checkpoint("[Radio emission]")

        # ===================== RADIO DETECTION ========================

        with timewith.TimeWith(
            "[RadioDetection]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Determine which stars fall into the sky region covered by the surveys.
            coverage_PMPS = survey_PMPS.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_SMPS = survey_SMPS.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_HTRU_low = survey_HTRU_low.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_HTRU_mid = survey_HTRU_mid.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_HTRU_high = survey_HTRU_high.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_SKA_low_AAstar = survey_SKA_low_AAstar.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_SKA_low_AA4 = survey_SKA_low_AA4.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_SKA_mid_band1_AAstar = (
                survey_SKA_mid_band1_AAstar.sky_coverage(
                    RA_final, DEC_final, l_final, b_final
                )
            )
            coverage_SKA_mid_band2_AAstar = (
                survey_SKA_mid_band2_AAstar.sky_coverage(
                    RA_final, DEC_final, l_final, b_final
                )
            )
            coverage_SKA_mid_band1_AA4 = survey_SKA_mid_band1_AA4.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )
            coverage_SKA_mid_band2_AA4 = survey_SKA_mid_band2_AA4.sky_coverage(
                RA_final, DEC_final, l_final, b_final
            )

            dist_cutoff = d_final < 35.0

            # Determine which stars fall into the sky region covered by any of the considered radio surveys.
            coverage_tot = (
                coverage_PMPS
                | coverage_SMPS
                | coverage_HTRU_low
                | coverage_HTRU_mid
                | coverage_HTRU_high
                | coverage_SKA_low_AAstar
                | coverage_SKA_low_AA4
                | coverage_SKA_mid_band1_AAstar
                | coverage_SKA_mid_band2_AAstar
                | coverage_SKA_mid_band1_AA4
                | coverage_SKA_mid_band2_AA4
            ) & dist_cutoff

            fraction_coverage = np.count_nonzero(coverage_tot) / NS_number
            log.info(
                f"Fraction of pulsars in the covered sky region: {fraction_coverage}"
            )

            timer.checkpoint("[Total sky coverage]")

            # Simulating the PMPS survey.
            log.info("Simulate detection with PMPS...")

            (
                detected_radio_PMPS,
                S_radio_obs_mean_PMPS,
                w_eff_PMPS,
                S_radio_obs_PMPS,
            ) = survey_PMPS.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_PMPS,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_PMPS = len(
                detected_radio_PMPS[detected_radio_PMPS]
            ) / len(detected_radio_PMPS)
            log.info(
                f"Fraction of detected pulsars by PMPS: {fraction_detected_radio_PMPS}"
            )

            # Simulating the SMPS survey.
            log.info("Simulate detection with SMPS...")

            (
                detected_radio_SMPS,
                S_radio_obs_mean_SMPS,
                w_eff_SMPS,
                S_radio_obs_SMPS,
            ) = survey_SMPS.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SMPS,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SMPS = len(
                detected_radio_SMPS[detected_radio_SMPS]
            ) / len(detected_radio_SMPS)
            log.info(
                f"Fraction of detected pulsars by SMPS: {fraction_detected_radio_SMPS}"
            )

            # Simulating the HTRU low-latitude survey.
            log.info("Simulate detection with HTRU low latitude...")

            (
                detected_radio_HTRU_low,
                S_radio_obs_mean_HTRU_low,
                w_eff_HTRU_low,
                S_radio_obs_HTRU_low,
            ) = survey_HTRU_low.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_HTRU_low,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_HTRU_low = len(
                detected_radio_HTRU_low[detected_radio_HTRU_low]
            ) / len(detected_radio_HTRU_low)

            log.info(
                f"Fraction of detected pulsars by HTRU low latitude: {fraction_detected_radio_HTRU_low}"
            )

            # Simulating the HTRU mid-latitude survey.
            log.info("Simulate detection with HTRU mid latitude...")

            (
                detected_radio_HTRU_mid,
                S_radio_obs_mean_HTRU_mid,
                w_eff_HTRU_mid,
                S_radio_obs_HTRU_mid,
            ) = survey_HTRU_mid.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_HTRU_mid,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_HTRU_mid = len(
                detected_radio_HTRU_mid[detected_radio_HTRU_mid]
            ) / len(detected_radio_HTRU_mid)

            log.info(
                f"Fraction of detected pulsars by HTRU mid latitude: {fraction_detected_radio_HTRU_mid}"
            )

            # Simulating the HTRU high-latitude survey.
            log.info("Simulate detection with HTRU high latitude...")

            (
                detected_radio_HTRU_high,
                S_radio_obs_mean_HTRU_high,
                w_eff_HTRU_high,
                S_radio_obs_HTRU_high,
            ) = survey_HTRU_high.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_HTRU_high,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_HTRU_high = len(
                detected_radio_HTRU_high[detected_radio_HTRU_high]
            ) / len(detected_radio_HTRU_high)

            log.info(
                f"Fraction of detected pulsars by HTRU high latitude: {fraction_detected_radio_HTRU_high}"
            )

            timer.checkpoint("[Radio surveys detection]")

            overlap_low_mid = detected_radio_HTRU_low & detected_radio_HTRU_mid

            detected_radio_HTRU_mid = (
                ~overlap_low_mid & detected_radio_HTRU_mid
            )

            # Simulating the SKA low survey.
            log.info("Simulate detection with SKA low...")

            (
                detected_radio_SKA_low_AAstar,
                S_radio_obs_mean_SKA_low_AAstar,
                w_eff_SKA_low_AAstar,
                S_radio_obs_SKA_low_AAstar,
            ) = survey_SKA_low_AAstar.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_low_AAstar,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_low_AAstar = len(
                detected_radio_SKA_low_AAstar[detected_radio_SKA_low_AAstar]
            ) / len(detected_radio_SKA_low_AAstar)
            log.info(
                f"Fraction of detected pulsars by SKA low AAstar: {fraction_detected_radio_SKA_low_AAstar}"
            )

            (
                detected_radio_SKA_low_AA4,
                S_radio_obs_mean_SKA_low_AA4,
                w_eff_SKA_low_AA4,
                S_radio_obs_SKA_low_AA4,
            ) = survey_SKA_low_AA4.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_low_AA4,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_low_AA4 = len(
                detected_radio_SKA_low_AA4[detected_radio_SKA_low_AA4]
            ) / len(detected_radio_SKA_low_AA4)
            log.info(
                f"Fraction of detected pulsars by SKA low AA4: {fraction_detected_radio_SKA_low_AA4}"
            )

            # Simulating the SKA mid survey.
            log.info("Simulate detection with SKA mid...")

            (
                detected_radio_SKA_mid_band1_AAstar,
                S_radio_obs_mean_SKA_mid_band1_AAstar,
                w_eff_SKA_mid_band1_AAstar,
                S_radio_obs_mid_band1_AAstar,
            ) = survey_SKA_mid_band1_AAstar.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_mid_band1_AAstar,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_mid_band1_AAstar = len(
                detected_radio_SKA_mid_band1_AAstar[
                    detected_radio_SKA_mid_band1_AAstar
                ]
            ) / len(detected_radio_SKA_mid_band1_AAstar)
            log.info(
                f"Fraction of detected pulsars by SKA mid band1 AAstar: {fraction_detected_radio_SKA_mid_band1_AAstar}"
            )

            (
                detected_radio_SKA_mid_band2_AAstar,
                S_radio_obs_mean_SKA_mid_band2_AAstar,
                w_eff_SKA_mid_band2_AAstar,
                S_radio_obs_mid_band2_AAstar,
            ) = survey_SKA_mid_band2_AAstar.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_mid_band2_AAstar,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_mid_band2_AAstar = len(
                detected_radio_SKA_mid_band2_AAstar[
                    detected_radio_SKA_mid_band2_AAstar
                ]
            ) / len(detected_radio_SKA_mid_band2_AAstar)
            log.info(
                f"Fraction of detected pulsars by SKA mid band2 AAstar: {fraction_detected_radio_SKA_mid_band2_AAstar}"
            )

            (
                detected_radio_SKA_mid_band1_AA4,
                S_radio_obs_mean_SKA_mid_band1_AA4,
                w_eff_SKA_mid_band1_AA4,
                S_radio_obs_mid_band1_AA4,
            ) = survey_SKA_mid_band1_AA4.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_mid_band1_AA4,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_mid_band1_AA4 = len(
                detected_radio_SKA_mid_band1_AA4[
                    detected_radio_SKA_mid_band1_AA4
                ]
            ) / len(detected_radio_SKA_mid_band1_AA4)
            log.info(
                f"Fraction of detected pulsars by SKA mid band1 AA4: {fraction_detected_radio_SKA_mid_band1_AA4}"
            )

            (
                detected_radio_SKA_mid_band2_AA4,
                S_radio_obs_mean_SKA_mid_band2_AA4,
                w_eff_SKA_mid_band2_AA4,
                S_radio_obs_mid_band2_AA4,
            ) = survey_SKA_mid_band2_AA4.detected_radio_population_full(
                w_int_final,
                DM_final,
                P_final,
                l_final,
                b_final,
                DEC_final,
                S_radio_bol_final,
                intercepted_radio_final,
                coverage_SKA_mid_band2_AA4,
                dist_cutoff,
                spectral_index_final,
                tau_sc_final,
            )

            fraction_detected_radio_SKA_mid_band2_AA4 = len(
                detected_radio_SKA_mid_band2_AA4[
                    detected_radio_SKA_mid_band2_AA4
                ]
            ) / len(detected_radio_SKA_mid_band2_AA4)
            log.info(
                f"Fraction of detected pulsars by SKA mid band2 AA4: {fraction_detected_radio_SKA_mid_band2_AA4}"
            )

            NS_idx_PMPS = NS_idx[detected_radio_PMPS]
            NS_idx_SMPS = NS_idx[detected_radio_SMPS]
            NS_idx_HTRU_low = NS_idx[detected_radio_HTRU_low]
            NS_idx_HTRU_mid = NS_idx[detected_radio_HTRU_mid]
            NS_idx_HTRU_high = NS_idx[detected_radio_HTRU_high]
            NS_idx_SKA_low_AAstar = NS_idx[detected_radio_SKA_low_AAstar]
            NS_idx_SKA_low_AA4 = NS_idx[detected_radio_SKA_low_AA4]
            NS_idx_SKA_mid_band1_AAstar = NS_idx[
                detected_radio_SKA_mid_band1_AAstar
            ]
            NS_idx_SKA_mid_band2_AAstar = NS_idx[
                detected_radio_SKA_mid_band2_AAstar
            ]
            NS_idx_SKA_mid_band1_AA4 = NS_idx[detected_radio_SKA_mid_band1_AA4]
            NS_idx_SKA_mid_band2_AA4 = NS_idx[detected_radio_SKA_mid_band2_AA4]

        # ===================== EXPORT OUTPUT ========================

        # Adding the final output to a data frame for export.
        log.info("Creating data frame for exporting...")

        # Exporting the population file containing the observed properties of neutron stars detected by PMPS.

        # Generating two header lines and merging them using MultiIndex.
        parameters = [
            "NS_idx",
            "RA",
            "DEC",
            "l",
            "b",
            "d",
            "DM",
            "pm_RA",
            "pm_DEC",
            "P",
            "P_dot",
            "S_radio_obs_mean",
            "w_eff",
        ]
        units = [
            " ",
            "[deg]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[kpc]",
            "[pc cm^-3]",
            "[mas yr^-1]",
            "[mas yr^-1]",
            "[s]",
            "[s s^-1]",
            "[Jy]",
            "[s]",
        ]

        header = pd.MultiIndex.from_arrays([parameters, units])

        df_PMPS = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_PMPS,
                    RA_final[NS_idx_PMPS],
                    DEC_final[NS_idx_PMPS],
                    l_final[NS_idx_PMPS],
                    b_final[NS_idx_PMPS],
                    d_final[NS_idx_PMPS],
                    DM_final[NS_idx_PMPS],
                    pm_RA_final[NS_idx_PMPS],
                    pm_DEC_final[NS_idx_PMPS],
                    P_final[NS_idx_PMPS],
                    P_dot_final[NS_idx_PMPS],
                    S_radio_obs_mean_PMPS[NS_idx_PMPS],
                    w_eff_PMPS[NS_idx_PMPS],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        PMPS_output_path = pathlib.Path().joinpath(
            output_path, "survey_PMPS_results.pkl.gz"
        )
        df_PMPS.to_pickle(PMPS_output_path, compression="gzip")

        log.info(
            f"Output of the PMPS survey generated in {os.getcwd()}/{PMPS_output_path}"
        )

        # Exporting the population file containing the observed properties of neutron stars detected by SMPS.

        df_SMPS = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SMPS,
                    RA_final[NS_idx_SMPS],
                    DEC_final[NS_idx_SMPS],
                    l_final[NS_idx_SMPS],
                    b_final[NS_idx_SMPS],
                    d_final[NS_idx_SMPS],
                    DM_final[NS_idx_SMPS],
                    pm_RA_final[NS_idx_SMPS],
                    pm_DEC_final[NS_idx_SMPS],
                    P_final[NS_idx_SMPS],
                    P_dot_final[NS_idx_SMPS],
                    S_radio_obs_mean_SMPS[NS_idx_SMPS],
                    w_eff_SMPS[NS_idx_SMPS],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SMPS_output_path = pathlib.Path().joinpath(
            output_path, "survey_SMPS_results.pkl.gz"
        )
        df_SMPS.to_pickle(SMPS_output_path, compression="gzip")

        log.info(
            f"Output of the SMPS survey generated in {os.getcwd()}/{SMPS_output_path}"
        )

        # Exporting the population file containing the observed properties of neutron stars detected by HTRU high.

        df_HTRU_high = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_HTRU_high,
                    RA_final[NS_idx_HTRU_high],
                    DEC_final[NS_idx_HTRU_high],
                    l_final[NS_idx_HTRU_high],
                    b_final[NS_idx_HTRU_high],
                    d_final[NS_idx_HTRU_high],
                    DM_final[NS_idx_HTRU_high],
                    pm_RA_final[NS_idx_HTRU_high],
                    pm_DEC_final[NS_idx_HTRU_high],
                    P_final[NS_idx_HTRU_high],
                    P_dot_final[NS_idx_HTRU_high],
                    S_radio_obs_mean_HTRU_high[NS_idx_HTRU_high],
                    w_eff_HTRU_high[NS_idx_HTRU_high],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        HTRU_high_output_path = pathlib.Path().joinpath(
            output_path, "survey_HTRU_high_results.pkl.gz"
        )
        df_HTRU_high.to_pickle(HTRU_high_output_path, compression="gzip")

        log.info(
            f"Output of the HTRU high-latitude survey generated in {os.getcwd()}/{HTRU_high_output_path}"
        )

        # Exporting the population file containing the observed properties of neutron stars detected by HTRU low and mid.

        df_HTRU_low = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_HTRU_low,
                    RA_final[NS_idx_HTRU_low],
                    DEC_final[NS_idx_HTRU_low],
                    l_final[NS_idx_HTRU_low],
                    b_final[NS_idx_HTRU_low],
                    d_final[NS_idx_HTRU_low],
                    DM_final[NS_idx_HTRU_low],
                    pm_RA_final[NS_idx_HTRU_low],
                    pm_DEC_final[NS_idx_HTRU_low],
                    P_final[NS_idx_HTRU_low],
                    P_dot_final[NS_idx_HTRU_low],
                    S_radio_obs_mean_HTRU_low[NS_idx_HTRU_low],
                    w_eff_HTRU_low[NS_idx_HTRU_low],
                ]
            ).T,
            columns=header,
        )

        df_HTRU_low["detected_HTRU_low"] = np.ones(len(df_HTRU_low))

        df_HTRU_mid = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_HTRU_mid,
                    RA_final[NS_idx_HTRU_mid],
                    DEC_final[NS_idx_HTRU_mid],
                    l_final[NS_idx_HTRU_mid],
                    b_final[NS_idx_HTRU_mid],
                    d_final[NS_idx_HTRU_mid],
                    DM_final[NS_idx_HTRU_mid],
                    pm_RA_final[NS_idx_HTRU_mid],
                    pm_DEC_final[NS_idx_HTRU_mid],
                    P_final[NS_idx_HTRU_mid],
                    P_dot_final[NS_idx_HTRU_mid],
                    S_radio_obs_mean_HTRU_mid[NS_idx_HTRU_mid],
                    w_eff_HTRU_mid[NS_idx_HTRU_mid],
                ]
            ).T,
            columns=header,
        )

        df_HTRU_mid["detected_HTRU_low"] = np.zeros(len(df_HTRU_mid))
        df_HTRU_low_mid = pd.concat([df_HTRU_low, df_HTRU_mid], axis=0)

        # Save the data frame as a compressed binary file.
        HTRU_low_mid_output_path = pathlib.Path().joinpath(
            output_path, "survey_HTRU_low_mid_results.pkl.gz"
        )
        df_HTRU_low_mid.to_pickle(HTRU_low_mid_output_path, compression="gzip")

        log.info(
            f"Output of the HTRU low and mid-latitude surveys generated in {os.getcwd()}/{HTRU_low_mid_output_path}"
        )

        # Exporting the population file containing the observed properties of neutron stars detected by SKA low.

        df_SKA_low_AAstar = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_low_AAstar,
                    RA_final[NS_idx_SKA_low_AAstar],
                    DEC_final[NS_idx_SKA_low_AAstar],
                    l_final[NS_idx_SKA_low_AAstar],
                    b_final[NS_idx_SKA_low_AAstar],
                    d_final[NS_idx_SKA_low_AAstar],
                    DM_final[NS_idx_SKA_low_AAstar],
                    pm_RA_final[NS_idx_SKA_low_AAstar],
                    pm_DEC_final[NS_idx_SKA_low_AAstar],
                    P_final[NS_idx_SKA_low_AAstar],
                    P_dot_final[NS_idx_SKA_low_AAstar],
                    S_radio_obs_mean_SKA_low_AAstar[NS_idx_SKA_low_AAstar],
                    w_eff_SKA_low_AAstar[NS_idx_SKA_low_AAstar],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_low_AAstar_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_low_AAstar_results.pkl.gz"
        )
        df_SKA_low_AAstar.to_pickle(
            SKA_low_AAstar_output_path, compression="gzip"
        )

        log.info(
            f"Output of the SKA low AAstar survey generated in {os.getcwd()}/{SKA_low_AAstar_output_path}"
        )

        df_SKA_low_AA4 = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_low_AA4,
                    RA_final[NS_idx_SKA_low_AA4],
                    DEC_final[NS_idx_SKA_low_AA4],
                    l_final[NS_idx_SKA_low_AA4],
                    b_final[NS_idx_SKA_low_AA4],
                    d_final[NS_idx_SKA_low_AA4],
                    DM_final[NS_idx_SKA_low_AA4],
                    pm_RA_final[NS_idx_SKA_low_AA4],
                    pm_DEC_final[NS_idx_SKA_low_AA4],
                    P_final[NS_idx_SKA_low_AA4],
                    P_dot_final[NS_idx_SKA_low_AA4],
                    S_radio_obs_mean_SKA_low_AA4[NS_idx_SKA_low_AA4],
                    w_eff_SKA_low_AA4[NS_idx_SKA_low_AA4],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_low_AA4_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_low_AA4_results.pkl.gz"
        )
        df_SKA_low_AA4.to_pickle(SKA_low_AA4_output_path, compression="gzip")

        log.info(
            f"Output of the SKA low AA4 survey generated in {os.getcwd()}/{SKA_low_AA4_output_path}"
        )

        # Exporting the population file containing the observed properties of neutron stars detected by SKA mid.

        df_SKA_mid_band1_AAstar = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_mid_band1_AAstar,
                    RA_final[NS_idx_SKA_mid_band1_AAstar],
                    DEC_final[NS_idx_SKA_mid_band1_AAstar],
                    l_final[NS_idx_SKA_mid_band1_AAstar],
                    b_final[NS_idx_SKA_mid_band1_AAstar],
                    d_final[NS_idx_SKA_mid_band1_AAstar],
                    DM_final[NS_idx_SKA_mid_band1_AAstar],
                    pm_RA_final[NS_idx_SKA_mid_band1_AAstar],
                    pm_DEC_final[NS_idx_SKA_mid_band1_AAstar],
                    P_final[NS_idx_SKA_mid_band1_AAstar],
                    P_dot_final[NS_idx_SKA_mid_band1_AAstar],
                    S_radio_obs_mean_SKA_mid_band1_AAstar[
                        NS_idx_SKA_mid_band1_AAstar
                    ],
                    w_eff_SKA_mid_band1_AAstar[NS_idx_SKA_mid_band1_AAstar],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_mid_band1_AAstar_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_mid_band1_AAstar_results.pkl.gz"
        )
        df_SKA_mid_band1_AAstar.to_pickle(
            SKA_mid_band1_AAstar_output_path, compression="gzip"
        )

        log.info(
            f"Output of the SKA mid band1 AAstar survey generated in {os.getcwd()}/{SKA_mid_band1_AAstar_output_path}"
        )

        df_SKA_mid_band2_AAstar = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_mid_band2_AAstar,
                    RA_final[NS_idx_SKA_mid_band2_AAstar],
                    DEC_final[NS_idx_SKA_mid_band2_AAstar],
                    l_final[NS_idx_SKA_mid_band2_AAstar],
                    b_final[NS_idx_SKA_mid_band2_AAstar],
                    d_final[NS_idx_SKA_mid_band2_AAstar],
                    DM_final[NS_idx_SKA_mid_band2_AAstar],
                    pm_RA_final[NS_idx_SKA_mid_band2_AAstar],
                    pm_DEC_final[NS_idx_SKA_mid_band2_AAstar],
                    P_final[NS_idx_SKA_mid_band2_AAstar],
                    P_dot_final[NS_idx_SKA_mid_band2_AAstar],
                    S_radio_obs_mean_SKA_mid_band2_AAstar[
                        NS_idx_SKA_mid_band2_AAstar
                    ],
                    w_eff_SKA_mid_band2_AAstar[NS_idx_SKA_mid_band2_AAstar],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_mid_band2_AAstar_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_mid_band2_AAstar_results.pkl.gz"
        )
        df_SKA_mid_band2_AAstar.to_pickle(
            SKA_mid_band2_AAstar_output_path, compression="gzip"
        )

        log.info(
            f"Output of the SKA mid band2 AAstar survey generated in {os.getcwd()}/{SKA_mid_band2_AAstar_output_path}"
        )

        df_SKA_mid_band1_AA4 = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_mid_band1_AA4,
                    RA_final[NS_idx_SKA_mid_band1_AA4],
                    DEC_final[NS_idx_SKA_mid_band1_AA4],
                    l_final[NS_idx_SKA_mid_band1_AA4],
                    b_final[NS_idx_SKA_mid_band1_AA4],
                    d_final[NS_idx_SKA_mid_band1_AA4],
                    DM_final[NS_idx_SKA_mid_band1_AA4],
                    pm_RA_final[NS_idx_SKA_mid_band1_AA4],
                    pm_DEC_final[NS_idx_SKA_mid_band1_AA4],
                    P_final[NS_idx_SKA_mid_band1_AA4],
                    P_dot_final[NS_idx_SKA_mid_band1_AA4],
                    S_radio_obs_mean_SKA_mid_band1_AA4[
                        NS_idx_SKA_mid_band1_AA4
                    ],
                    w_eff_SKA_mid_band1_AA4[NS_idx_SKA_mid_band1_AA4],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_mid_band1_AA4_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_mid_band1_AA4_results.pkl.gz"
        )
        df_SKA_mid_band1_AA4.to_pickle(
            SKA_mid_band1_AA4_output_path, compression="gzip"
        )

        log.info(
            f"Output of the SKA mid band1 AA4 survey generated in {os.getcwd()}/{SKA_mid_band1_AA4_output_path}"
        )

        df_SKA_mid_band2_AA4 = pd.DataFrame(
            data=np.array(
                [
                    NS_idx_SKA_mid_band2_AA4,
                    RA_final[NS_idx_SKA_mid_band2_AA4],
                    DEC_final[NS_idx_SKA_mid_band2_AA4],
                    l_final[NS_idx_SKA_mid_band2_AA4],
                    b_final[NS_idx_SKA_mid_band2_AA4],
                    d_final[NS_idx_SKA_mid_band2_AA4],
                    DM_final[NS_idx_SKA_mid_band2_AA4],
                    pm_RA_final[NS_idx_SKA_mid_band2_AA4],
                    pm_DEC_final[NS_idx_SKA_mid_band2_AA4],
                    P_final[NS_idx_SKA_mid_band2_AA4],
                    P_dot_final[NS_idx_SKA_mid_band2_AA4],
                    S_radio_obs_mean_SKA_mid_band2_AA4[
                        NS_idx_SKA_mid_band2_AA4
                    ],
                    w_eff_SKA_mid_band2_AA4[NS_idx_SKA_mid_band2_AA4],
                ]
            ).T,
            columns=header,
        )

        # Save the data frame as a compressed binary file.
        SKA_mid_band2_AA4_output_path = pathlib.Path().joinpath(
            output_path, "survey_SKA_mid_band2_AA4_results.pkl.gz"
        )
        df_SKA_mid_band2_AA4.to_pickle(
            SKA_mid_band2_AA4_output_path, compression="gzip"
        )

        log.info(
            f"Output of the SKA mid band2 AA4 survey generated in {os.getcwd()}/{SKA_mid_band2_AA4_output_path}"
        )

        timer.checkpoint("[Export]")

    # Cleanup. Reset seed to empty value.
    cfg["seed_full"] = None


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="output/sim_full_survey_only",
        help="Path to the directory where the run will be saved.",
    )

    args.add_argument(
        "--full_data",
        nargs="?",
        type=str,
        default="output/sim_dyn",
        help="Path to the location where a final_population.pkl.gz file is saved.",
    )

    args = args.parse_args()

    simulate_surveys(args)
