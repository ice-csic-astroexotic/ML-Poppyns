"""
Simulating a detected population of neutron stars from a dynamically evolved population database.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)

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

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.basics.random_sampler as rs
import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
from pypopsyn.simulator.configuration import cfg
from scripts.memory_efficient_sampling import select

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def simulate_population(args) -> None:
    """
    Simulating a detected neutron star population starting from a dynamically evolved
    population database.

    Args:

        dyn_data (str): Path to a dynamically evolved population database.
        output_path (str): Output directory for the run.
        parameter_override (str): Path to JSON with parameter overrides.

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

    # Check if the parsed dynamically simulated population directory exists.
    dyn_path = pathlib.Path(args.dyn_data)
    dyn_path_config = pathlib.Path().joinpath(dyn_path, "configuration.json")
    dyn_path_pop = pathlib.Path().joinpath(dyn_path, "final_pop_dyn.csv")

    if not dyn_path_pop.exists():
        log.error(f"File {dyn_path} not found...")
        sys.exit()

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

    # Loading the sever path, in case simulations are run at the PIC.
    path_server_software = cfg["path_server_software"]

    # Initialize the surveys.
    PMPS_par_path = pathlib.Path().joinpath(
        path_server_software,
        "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
    )

    SMPS_par_path = pathlib.Path().joinpath(
        path_server_software,
        "pypopsyn/simulator/multiband_surveys/Swinburne_Parkes_parameters.json",
    )

    HTRU_low_par_path = pathlib.Path().joinpath(
        path_server_software,
        "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
    )

    HTRU_mid_par_path = pathlib.Path().joinpath(
        path_server_software,
        "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
    )

    HTRU_high_par_path = pathlib.Path().joinpath(
        path_server_software,
        "pypopsyn/simulator/multiband_surveys/htru_high_parameters.json",
    )

    survey_PMPS = sr.SurveyRadio(PMPS_par_path)
    survey_SMPS = sr.SurveyRadio(SMPS_par_path)
    survey_HTRU_low = sr.SurveyRadio(HTRU_low_par_path)
    survey_HTRU_mid = sr.SurveyRadio(HTRU_mid_par_path)
    survey_HTRU_high = sr.SurveyRadio(HTRU_high_par_path)

    n_detected_real_PMPS = cfg["detected_real_PMPS"]
    n_detected_real_SMPS = cfg["detected_real_SMPS"]
    n_detected_real_HTRU_low_mid = cfg["detected_real_htru_low_mid"]
    n_detected_real_HTRU_high = cfg["detected_real_htru_high"]

    # Import the maximum age value from the dynamical configuration file.
    # This is needed for the computation of the birth rate.
    with open(dyn_path_config, "r") as f:
        conf_json = json.load(f)

    t_max = conf_json["t_age_max"] / 100  # Maximum time in centuries.

    # Initialize the indicator for an excess in birth rate to False.
    # This variable will be changed to True if the birth rate exceeds a value of 5 NS / century.
    # It is finally saved in the configuration.json file where the simulation output is stored.
    cfg["birth_rate_excess"] = False

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):
        # ===================== EVOLVE AND DETECT ========================

        # These variables count how many stars we create to reach the desirable number in each survey.
        n_created = 0
        n_created_PMPS = 0
        n_created_SMPS = 0
        n_created_HTRU_low_mid = 0
        n_created_HTRU_high = 0

        n_detected_sim_PMPS = 0
        n_detected_sim_SMPS = 0
        n_detected_sim_HTRU_low = 0
        n_detected_sim_HTRU_mid = 0
        n_detected_sim_HTRU_high = 0
        n_detected_sim_HTRU_low_mid = 0

        stop_PMPS = False
        stop_SMPS = False
        stop_HTRU_low_mid = False
        stop_HTRU_high = False

        # Define a list to store the indices of the detected neutron stars to avoid resampling.
        # In the first iteration, we are not removing any indices.
        idx_remove = []

        # To speed up the simulation, generate new neutron stars in batches.
        n_batchsize = 100000

        # Once we have detected 90% or 95% of the NS in both surveys, we reduce the batch size to speed up the
        # simulations. We initialize these flags as false.
        flag_90 = False
        flag_95 = False

        # Initializing the dictionaries where we save the detected neutron stars for each survey.
        dictionary_detected_PMPS = {
            "age": [],
            "ra": [],
            "dec": [],
            "l": [],
            "b": [],
            "DM": [],
            "dist": [],
            "pm_ra": [],
            "pm_dec": [],
            "v_ls": [],
            "B": [],
            "chi": [],
            "P": [],
            "Pdot": [],
            "L_radio_bol": [],
            "S_radio_obs_mean": [],
            "w_int": [],
            "w_eff": [],
        }

        dictionary_detected_SMPS = {
            "age": [],
            "ra": [],
            "dec": [],
            "l": [],
            "b": [],
            "DM": [],
            "dist": [],
            "pm_ra": [],
            "pm_dec": [],
            "v_ls": [],
            "B": [],
            "chi": [],
            "P": [],
            "Pdot": [],
            "L_radio_bol": [],
            "S_radio_obs_mean": [],
            "w_int": [],
            "w_eff": [],
        }

        dictionary_detected_HTRU_low = {
            "age": [],
            "ra": [],
            "dec": [],
            "l": [],
            "b": [],
            "DM": [],
            "dist": [],
            "pm_ra": [],
            "pm_dec": [],
            "v_ls": [],
            "B": [],
            "chi": [],
            "P": [],
            "Pdot": [],
            "L_radio_bol": [],
            "S_radio_obs_mean": [],
            "w_int": [],
            "w_eff": [],
            "HTRU_low": [],
            "HTRU_mid": [],
            "HTRU_high": [],
        }

        dictionary_detected_HTRU_mid = {
            "age": [],
            "ra": [],
            "dec": [],
            "l": [],
            "b": [],
            "DM": [],
            "dist": [],
            "pm_ra": [],
            "pm_dec": [],
            "v_ls": [],
            "B": [],
            "chi": [],
            "P": [],
            "Pdot": [],
            "L_radio_bol": [],
            "S_radio_obs_mean": [],
            "w_int": [],
            "w_eff": [],
            "HTRU_low": [],
            "HTRU_mid": [],
            "HTRU_high": [],
        }

        dictionary_detected_HTRU_high = {
            "age": [],
            "ra": [],
            "dec": [],
            "l": [],
            "b": [],
            "DM": [],
            "dist": [],
            "pm_ra": [],
            "pm_dec": [],
            "v_ls": [],
            "B": [],
            "chi": [],
            "P": [],
            "Pdot": [],
            "L_radio_bol": [],
            "S_radio_obs_mean": [],
            "w_int": [],
            "w_eff": [],
            "HTRU_low": [],
            "HTRU_mid": [],
            "HTRU_high": [],
        }

        # Continue to simulate stars until the detected number of pulsars for all the surveys is reached.
        while (
            (n_detected_sim_PMPS < n_detected_real_PMPS)
            | (n_detected_sim_SMPS < n_detected_real_SMPS)
            | (n_detected_sim_HTRU_low_mid < n_detected_real_HTRU_low_mid)
            | (n_detected_sim_HTRU_high < n_detected_real_HTRU_high)
        ):

            with timewith.TimeWith(
                "[LoadPopulationDynamics]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                # ===================== INITIALIZE THE POPULATION ========================

                # Evaluate the percentage of neutron stars detected by the simulated
                # surveys with respect to the real surveys.
                percentage_detected_PMPS = (
                    n_detected_sim_PMPS / n_detected_real_PMPS
                )
                percentage_detected_SMPS = (
                    n_detected_sim_SMPS / n_detected_real_SMPS
                )
                percentage_detected_HTRU_low_mid = (
                    n_detected_sim_HTRU_low_mid
                ) / n_detected_real_HTRU_low_mid

                percentage_detected_HTRU_high = (
                    n_detected_sim_HTRU_high
                ) / n_detected_real_HTRU_high

                # If the percentage of both surveys is over 90% reduce the batch size.
                if (
                    (percentage_detected_PMPS > 0.9)
                    & (percentage_detected_SMPS > 0.9)
                    & (percentage_detected_HTRU_low_mid > 0.9)
                    & (percentage_detected_HTRU_high > 0.9)
                    & (flag_90 is False)
                ):
                    flag_90 = True
                    n_batchsize = 10000

                elif (
                    (percentage_detected_PMPS > 0.95)
                    & (percentage_detected_SMPS > 0.95)
                    & (percentage_detected_HTRU_low_mid > 0.95)
                    & (percentage_detected_HTRU_high > 0.95)
                    & (flag_95 is False)
                ):
                    flag_95 = True
                    n_batchsize = 5000

                # Update the total number of simulated neutron stars.
                n_created += n_batchsize

                log.info(f"Total number of created neutron stars: {n_created}")

                # Load the chunk of the file containing the dynamically evolved population parameters.
                df_dyn = select(
                    dyn_path_pop,
                    n_batchsize,
                    conf_json["NS_number"],
                    idx_remove,
                )

                age = df_dyn["age"]["[yr]"].to_numpy()
                r_final = df_dyn["r"]["[kpc]"].to_numpy()
                phi_final = df_dyn["phi"]["[rad]"].to_numpy()
                z_final = df_dyn["z"]["[kpc]"].to_numpy()
                v_r_final = df_dyn["v_r"]["[km/s]"].to_numpy()
                v_phi_final = df_dyn["v_phi"]["[km/s]"].to_numpy()
                v_z_final = df_dyn["v_z"]["[km/s]"].to_numpy()

                # Convert from polar coordinates to Cartesian coordinates.
                x_final, y_final = coco.polar_to_cartesian(r_final, phi_final)

                # Convert velocity components from galactocentric cylindrical coordinates
                # to galactocentric Cartesian coordinates.
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
                    sun_dist_icrs,
                    pm_ra_final,
                    pm_dec_final,
                    v_ls_icrs,
                ) = coco.galactocentric_to_icrs(
                    x_final, y_final, z_final, v_x_final, v_y_final, v_z_final
                )

                # Convert galactocentric coordinates and velocities into galactic coordinates.
                (
                    l_final,
                    b_final,
                    sun_dist_gal,
                    pm_l_final,
                    pm_b_final,
                    v_ls_gal,
                ) = coco.galactocentric_to_galactic(
                    x_final, y_final, z_final, v_x_final, v_y_final, v_z_final
                )

            with timewith.TimeWith(
                "[SimulatePopulationDetection]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                dist_cutoff = sun_dist_icrs < 35.0
                age_datab = age
                ra_datab = ra_final
                dec_datab = dec_final
                l_datab = l_final
                b_datab = b_final
                dist_datab = sun_dist_icrs

                # Select only neutron stars that fall into the sky region covered by the surveys.
                coverage_PMPS = survey_PMPS.sky_coverage(
                    ra_datab, dec_datab, l_datab, b_datab
                )
                coverage_SMPS = survey_SMPS.sky_coverage(
                    ra_datab, dec_datab, l_datab, b_datab
                )
                coverage_HTRU_low = survey_HTRU_low.sky_coverage(
                    ra_datab, dec_datab, l_datab, b_datab
                )
                coverage_HTRU_mid = survey_HTRU_mid.sky_coverage(
                    ra_datab, dec_datab, l_datab, b_datab
                )
                coverage_HTRU_high = survey_HTRU_high.sky_coverage(
                    ra_datab, dec_datab, l_datab, b_datab
                )

                # Saving the full dataset indices of the selected rows.
                idx = df_dyn.index.values
                idx_pos = np.arange(len(idx))
                df_index = pd.DataFrame(
                    data={"index position": idx_pos}, index=idx
                )

                # Determine which stars fall into the sky region covered by any of the considered radio surveys.
                coverage_tot = (
                    coverage_PMPS
                    | coverage_SMPS
                    | coverage_HTRU_low
                    | coverage_HTRU_mid
                    | coverage_HTRU_high
                ) & dist_cutoff

                idx_det = idx[coverage_tot]

                age_det = age_datab[coverage_tot]
                l_det = l_datab[coverage_tot]
                b_det = b_datab[coverage_tot]
                dist_det = dist_datab[coverage_tot]
                coverage_PMPS = coverage_PMPS[coverage_tot]
                coverage_SMPS = coverage_SMPS[coverage_tot]
                coverage_HTRU_low = coverage_HTRU_low[coverage_tot]
                coverage_HTRU_mid = coverage_HTRU_mid[coverage_tot]
                coverage_HTRU_high = coverage_HTRU_high[coverage_tot]

                # Remove stars that do not fall into the total sky coverage.
                out_coverage = np.invert(coverage_tot)
                idx_remove += idx[out_coverage].tolist()

                # Initialize neutron star population properties.
                pop_initial = ipop.InitialNeutronStarPopulation(
                    NS_number=len(age_det)
                )

                # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================

                # Computing the initial field strengths, misalignment angles, and periods.
                B_initial = pop_initial.magnetic_field()
                chi_initial = pop_initial.misalignment_angle()
                P_initial = pop_initial.period()

                # Determine the evolved magnetic field, misalignment angle and rotation period.
                (
                    B_det,
                    chi_det,
                    P_det,
                    NS_magrot_evol_dict,
                ) = mre.magneto_rotational_evolution(
                    B_initial,
                    chi_initial,
                    P_initial,
                    age_det,
                )

                # ===================== RADIO EMISSION ========================

                dictionary_detected = er.radio_emission(
                    P_det,
                    age_det,
                    l_det,
                    b_det,
                    dist_det,
                    B_det,
                    chi_det,
                    idx_det,
                )

                age_det = dictionary_detected["age_det"]
                l_det = dictionary_detected["l_det"]
                b_det = dictionary_detected["b_det"]
                B_det = dictionary_detected["B_det"]
                chi_det = dictionary_detected["chi_det"]
                P_det = dictionary_detected["P_det"]
                w_int_s = dictionary_detected["w_int_s"]
                DM = dictionary_detected["DM"]
                idx_det = dictionary_detected["idx_det"]
                L_radio_bol = dictionary_detected["L_radio_bol"]
                S_radio_bol = dictionary_detected["S_radio_bol"]
                P_dot_det = dictionary_detected["P_dot_det"]
                intercepted_radio = dictionary_detected["intercepted_radio"]

                coverage_PMPS = coverage_PMPS[intercepted_radio]
                coverage_SMPS = coverage_SMPS[intercepted_radio]
                coverage_HTRU_low = coverage_HTRU_low[intercepted_radio]
                coverage_HTRU_mid = coverage_HTRU_mid[intercepted_radio]
                coverage_HTRU_high = coverage_HTRU_high[intercepted_radio]

                if np.count_nonzero(intercepted_radio) == 0:
                    break

                # ===================== RADIO DETECTION ========================

                # ======== Simulating PMPS. ========

                (
                    detected_radio_PMPS,
                    w_eff_PMPS,
                    S_radio_obs_mean_PMPS,
                ) = survey_PMPS.radio_detection(
                    w_int_s,
                    DM,
                    P_det,
                    age_det,
                    coverage_PMPS,
                    l_det,
                    b_det,
                    S_radio_bol,
                )

                n_detected_sim_PMPS += np.count_nonzero(detected_radio_PMPS)
                log.info(
                    f"Total number of neutron stars detected by the Parkes Multibeam Survey: {n_detected_sim_PMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with PMPS is reached.
                # This is needed to compute the birth rate derived from the PMPS detections.
                if (n_detected_sim_PMPS >= n_detected_real_PMPS) & (
                    stop_PMPS is False
                ):
                    stop_PMPS = True
                    n_created_PMPS = n_created

                # ======== Simulating SMPS. ========

                (
                    detected_radio_SMPS,
                    w_eff_SMPS,
                    S_radio_obs_mean_SMPS,
                ) = survey_SMPS.radio_detection(
                    w_int_s,
                    DM,
                    P_det,
                    age_det,
                    coverage_SMPS,
                    l_det,
                    b_det,
                    S_radio_bol,
                )

                n_detected_sim_SMPS += np.count_nonzero(detected_radio_SMPS)
                log.info(
                    f"Total number of neutron stars detected by the Swinburne Parkes Multibeam Survey: {n_detected_sim_SMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with SMPS is reached.
                # This is needed to compute the birth rate derived from the SMPS detections.
                if (n_detected_sim_SMPS >= n_detected_real_SMPS) & (
                    stop_SMPS is False
                ):
                    stop_SMPS = True
                    n_created_SMPS = n_created

                # ======== Simulating the HTRU low survey. ========

                (
                    detected_radio_HTRU_low,
                    w_eff_HTRU_low,
                    S_radio_obs_mean_HTRU_low,
                ) = survey_HTRU_low.radio_detection(
                    w_int_s,
                    DM,
                    P_det,
                    age_det,
                    coverage_HTRU_low,
                    l_det,
                    b_det,
                    S_radio_bol,
                )

                n_detected_sim_HTRU_low += np.count_nonzero(
                    detected_radio_HTRU_low
                )

                # ======== Simulating the HTRU mid survey. ========

                (
                    detected_radio_HTRU_mid,
                    w_eff_HTRU_mid,
                    S_radio_obs_mean_HTRU_mid,
                ) = survey_HTRU_mid.radio_detection(
                    w_int_s,
                    DM,
                    P_det,
                    age_det,
                    coverage_HTRU_mid,
                    l_det,
                    b_det,
                    S_radio_bol,
                )

                # Since the sky coverage of the HTRU mid and low surveys overlap, we remove those stars from the mid
                # survey that are already in the low survey in order to not double count individual objects.
                overlap_low_mid = (
                    detected_radio_HTRU_low & detected_radio_HTRU_mid
                )

                detected_radio_HTRU_mid = (
                    ~overlap_low_mid & detected_radio_HTRU_mid
                )

                n_detected_sim_HTRU_mid += np.count_nonzero(
                    detected_radio_HTRU_mid
                )

                n_detected_sim_HTRU_low_mid = (
                    n_detected_sim_HTRU_low + n_detected_sim_HTRU_mid
                )

                log.info(
                    f"Total number of neutron stars detected by the HTRU mid and low latitude surveys: {n_detected_sim_HTRU_low_mid}"
                )

                # Store the value of created neutron stars once the number of detected pulsars with HTRU is reached.
                # This is needed to compute the birth rate derived from the HTRU detections.
                if (
                    n_detected_sim_HTRU_low_mid >= n_detected_real_HTRU_low_mid
                ) & (stop_HTRU_low_mid is False):
                    stop_HTRU_low_mid = True
                    n_created_HTRU_low_mid = n_created

                # ======== Simulating the HTRU high survey. ========

                (
                    detected_radio_HTRU_high,
                    w_eff_HTRU_high,
                    S_radio_obs_mean_HTRU_high,
                ) = survey_HTRU_high.radio_detection(
                    w_int_s,
                    DM,
                    P_det,
                    age_det,
                    coverage_HTRU_high,
                    l_det,
                    b_det,
                    S_radio_bol,
                )

                n_detected_sim_HTRU_high += np.count_nonzero(
                    detected_radio_HTRU_high
                )

                log.info(
                    f"Total number of neutron stars detected by the HTRU high latitude survey: {n_detected_sim_HTRU_high}"
                )

                # Store the value of created neutron stars once the number of detected pulsars with HTRU is reached.
                # This is needed to compute the birth rate derived from the HTRU detections.
                if (n_detected_sim_HTRU_high >= n_detected_real_HTRU_high) & (
                    stop_HTRU_high is False
                ):
                    stop_HTRU_high = True
                    n_created_HTRU_high = n_created

                # Select only neutron stars that are detected by one of the surveys.
                idx_det_PMPS = idx_det[detected_radio_PMPS]
                idx_det_SMPS = idx_det[detected_radio_SMPS]
                idx_det_HTRU_low = idx_det[detected_radio_HTRU_low]
                idx_det_HTRU_mid = idx_det[detected_radio_HTRU_mid]
                idx_det_HTRU_high = idx_det[detected_radio_HTRU_high]

                idx_det_PMPS = df_index["index position"][idx_det_PMPS].values
                idx_det_SMPS = df_index["index position"][idx_det_SMPS].values
                idx_det_HTRU_low = df_index["index position"][
                    idx_det_HTRU_low
                ].values
                idx_det_HTRU_mid = df_index["index position"][
                    idx_det_HTRU_mid
                ].values
                idx_det_HTRU_high = df_index["index position"][
                    idx_det_HTRU_high
                ].values

                # Update the database of detected neutron stars.
                update_dictionary_detected_PMPS = {
                    "age": age[idx_det_PMPS].tolist(),
                    "ra": ra_final[idx_det_PMPS].tolist(),
                    "dec": dec_final[idx_det_PMPS].tolist(),
                    "l": l_final[idx_det_PMPS].tolist(),
                    "b": b_final[idx_det_PMPS].tolist(),
                    "DM": DM[detected_radio_PMPS].tolist(),
                    "dist": sun_dist_icrs[idx_det_PMPS].tolist(),
                    "pm_ra": pm_ra_final[idx_det_PMPS].tolist(),
                    "pm_dec": pm_dec_final[idx_det_PMPS].tolist(),
                    "v_ls": v_ls_icrs[idx_det_PMPS].tolist(),
                    "B": B_det[detected_radio_PMPS].tolist(),
                    "chi": chi_det[detected_radio_PMPS].tolist(),
                    "P": P_det[detected_radio_PMPS].tolist(),
                    "Pdot": P_dot_det[detected_radio_PMPS].tolist(),
                    "L_radio_bol": L_radio_bol[detected_radio_PMPS].tolist(),
                    "S_radio_obs_mean": S_radio_obs_mean_PMPS[
                        detected_radio_PMPS
                    ].tolist(),
                    "w_int": w_int_s[detected_radio_PMPS].tolist(),
                    "w_eff": w_eff_PMPS[detected_radio_PMPS].tolist(),
                }

                # Update the dictionary containing the detection information.
                dictionary_detected_PMPS = {
                    key: value + update_dictionary_detected_PMPS[key]
                    for key, value in dictionary_detected_PMPS.items()
                }

                update_dictionary_detected_SMPS = {
                    "age": age[idx_det_SMPS].tolist(),
                    "ra": ra_final[idx_det_SMPS].tolist(),
                    "dec": dec_final[idx_det_SMPS].tolist(),
                    "l": l_final[idx_det_SMPS].tolist(),
                    "b": b_final[idx_det_SMPS].tolist(),
                    "DM": DM[detected_radio_SMPS].tolist(),
                    "dist": sun_dist_icrs[idx_det_SMPS].tolist(),
                    "pm_ra": pm_ra_final[idx_det_SMPS].tolist(),
                    "pm_dec": pm_dec_final[idx_det_SMPS].tolist(),
                    "v_ls": v_ls_icrs[idx_det_SMPS].tolist(),
                    "B": B_det[detected_radio_SMPS].tolist(),
                    "chi": chi_det[detected_radio_SMPS].tolist(),
                    "P": P_det[detected_radio_SMPS].tolist(),
                    "Pdot": P_dot_det[detected_radio_SMPS].tolist(),
                    "L_radio_bol": L_radio_bol[detected_radio_SMPS].tolist(),
                    "S_radio_obs_mean": S_radio_obs_mean_SMPS[
                        detected_radio_SMPS
                    ].tolist(),
                    "w_int": w_int_s[detected_radio_SMPS].tolist(),
                    "w_eff": w_eff_SMPS[detected_radio_SMPS].tolist(),
                }

                # Update the dictionary containing the detection information.
                dictionary_detected_SMPS = {
                    key: value + update_dictionary_detected_SMPS[key]
                    for key, value in dictionary_detected_SMPS.items()
                }

                type_survey_list = np.zeros(len(DM))
                type_survey_list[overlap_low_mid] = 1

                update_dictionary_detected_HTRU_low = {
                    "age": age[idx_det_HTRU_low].tolist(),
                    "ra": ra_final[idx_det_HTRU_low].tolist(),
                    "dec": dec_final[idx_det_HTRU_low].tolist(),
                    "l": l_final[idx_det_HTRU_low].tolist(),
                    "b": b_final[idx_det_HTRU_low].tolist(),
                    "DM": DM[detected_radio_HTRU_low].tolist(),
                    "dist": sun_dist_icrs[idx_det_HTRU_low].tolist(),
                    "pm_ra": pm_ra_final[idx_det_HTRU_low].tolist(),
                    "pm_dec": pm_dec_final[idx_det_HTRU_low].tolist(),
                    "v_ls": v_ls_icrs[idx_det_HTRU_low].tolist(),
                    "B": B_det[detected_radio_HTRU_low].tolist(),
                    "chi": chi_det[detected_radio_HTRU_low].tolist(),
                    "P": P_det[detected_radio_HTRU_low].tolist(),
                    "Pdot": P_dot_det[detected_radio_HTRU_low].tolist(),
                    "L_radio_bol": L_radio_bol[
                        detected_radio_HTRU_low
                    ].tolist(),
                    "S_radio_obs_mean": S_radio_obs_mean_HTRU_low[
                        detected_radio_HTRU_low
                    ].tolist(),
                    "w_int": w_int_s[detected_radio_HTRU_low].tolist(),
                    "w_eff": w_eff_HTRU_low[detected_radio_HTRU_low].tolist(),
                    "HTRU_low": np.ones(len(idx_det_HTRU_low)).tolist(),
                    "HTRU_mid": type_survey_list[
                        detected_radio_HTRU_low
                    ].tolist(),
                    "HTRU_high": np.zeros(len(idx_det_HTRU_low)).tolist(),
                }

                # Update the dictionary containing the detection information.
                dictionary_detected_HTRU_low = {
                    key: value + update_dictionary_detected_HTRU_low[key]
                    for key, value in dictionary_detected_HTRU_low.items()
                }

                update_dictionary_detected_HTRU_mid = {
                    "age": age[idx_det_HTRU_mid].tolist(),
                    "ra": ra_final[idx_det_HTRU_mid].tolist(),
                    "dec": dec_final[idx_det_HTRU_mid].tolist(),
                    "l": l_final[idx_det_HTRU_mid].tolist(),
                    "b": b_final[idx_det_HTRU_mid].tolist(),
                    "DM": DM[detected_radio_HTRU_mid].tolist(),
                    "dist": sun_dist_icrs[idx_det_HTRU_mid].tolist(),
                    "pm_ra": pm_ra_final[idx_det_HTRU_mid].tolist(),
                    "pm_dec": pm_dec_final[idx_det_HTRU_mid].tolist(),
                    "v_ls": v_ls_icrs[idx_det_HTRU_mid].tolist(),
                    "B": B_det[detected_radio_HTRU_mid].tolist(),
                    "chi": chi_det[detected_radio_HTRU_mid].tolist(),
                    "P": P_det[detected_radio_HTRU_mid].tolist(),
                    "Pdot": P_dot_det[detected_radio_HTRU_mid].tolist(),
                    "L_radio_bol": L_radio_bol[
                        detected_radio_HTRU_mid
                    ].tolist(),
                    "S_radio_obs_mean": S_radio_obs_mean_HTRU_mid[
                        detected_radio_HTRU_mid
                    ].tolist(),
                    "w_int": w_int_s[detected_radio_HTRU_mid].tolist(),
                    "w_eff": w_eff_HTRU_mid[detected_radio_HTRU_mid].tolist(),
                    "HTRU_low": np.zeros(len(idx_det_HTRU_mid)).tolist(),
                    "HTRU_mid": np.ones(len(idx_det_HTRU_mid)).tolist(),
                    "HTRU_high": np.zeros(len(idx_det_HTRU_mid)).tolist(),
                }

                # Update the dictionary containing the detection information.
                dictionary_detected_HTRU_mid = {
                    key: value + update_dictionary_detected_HTRU_mid[key]
                    for key, value in dictionary_detected_HTRU_mid.items()
                }

                update_dictionary_detected_HTRU_high = {
                    "age": age[idx_det_HTRU_high].tolist(),
                    "ra": ra_final[idx_det_HTRU_high].tolist(),
                    "dec": dec_final[idx_det_HTRU_high].tolist(),
                    "l": l_final[idx_det_HTRU_high].tolist(),
                    "b": b_final[idx_det_HTRU_high].tolist(),
                    "DM": DM[detected_radio_HTRU_high].tolist(),
                    "dist": sun_dist_icrs[idx_det_HTRU_high].tolist(),
                    "pm_ra": pm_ra_final[idx_det_HTRU_high].tolist(),
                    "pm_dec": pm_dec_final[idx_det_HTRU_high].tolist(),
                    "v_ls": v_ls_icrs[idx_det_HTRU_high].tolist(),
                    "B": B_det[detected_radio_HTRU_high].tolist(),
                    "chi": chi_det[detected_radio_HTRU_high].tolist(),
                    "P": P_det[detected_radio_HTRU_high].tolist(),
                    "Pdot": P_dot_det[detected_radio_HTRU_high].tolist(),
                    "L_radio_bol": L_radio_bol[
                        detected_radio_HTRU_high
                    ].tolist(),
                    "S_radio_obs_mean": S_radio_obs_mean_HTRU_high[
                        detected_radio_HTRU_high
                    ].tolist(),
                    "w_int": w_int_s[detected_radio_HTRU_high].tolist(),
                    "w_eff": w_eff_HTRU_high[
                        detected_radio_HTRU_high
                    ].tolist(),
                    "HTRU_low": np.zeros(len(idx_det_HTRU_high)).tolist(),
                    "HTRU_mid": np.zeros(len(idx_det_HTRU_high)).tolist(),
                    "HTRU_high": np.ones(len(idx_det_HTRU_high)).tolist(),
                }

                # Update the dictionary containing the detection information.
                dictionary_detected_HTRU_high = {
                    key: value + update_dictionary_detected_HTRU_high[key]
                    for key, value in dictionary_detected_HTRU_high.items()
                }

                # Remove from the dynamical database the stars that have been detected or
                # that are outside the sky coverage of the surveys.
                detected = (
                    detected_radio_PMPS
                    | detected_radio_SMPS
                    | detected_radio_HTRU_low
                    | detected_radio_HTRU_mid
                    | detected_radio_HTRU_high
                )
                idx_det_tot = idx_det[detected]
                idx_remove += idx_det_tot.tolist()

                # If the current birth rate exceeds an upper limit of 5 NS per century stop the simulation.
                birth_rate = n_created / t_max
                log.info(
                    f"Galactic neutron star birth rate per century: {birth_rate} neutron stars per century."
                )
                if birth_rate > 5:
                    log.info(
                        "Simulation stopped! Galactic neutron star birth rate per century exceeds 5 neutron stars per century."
                    )
                    n_created_PMPS = n_created
                    n_created_SMPS = n_created
                    n_created_HTRU_low_mid = n_created
                    n_created_HTRU_high = n_created

                    # Set the indicator of an excess in birth rate to True.
                    cfg["birth_rate_excess"] = True

                    break

        # Determine the Galactic neutron star birth rate per century for the different surveys.
        birth_rate_PMPS = n_created_PMPS / t_max
        birth_rate_SMPS = n_created_SMPS / t_max
        birth_rate_HTRU_low_mid = n_created_HTRU_low_mid / t_max
        birth_rate_HTRU_high = n_created_HTRU_high / t_max

        log.info(
            f"Galactic neutron star birth rate per century according to PMPS: {birth_rate_PMPS} neutron stars per century."
        )
        log.info(
            f"Galactic neutron star birth rate per century according to SMPS: {birth_rate_SMPS} neutron stars per century."
        )
        log.info(
            f"Galactic neutron star birth rate per century according to HTRU mid and low surveys: {birth_rate_HTRU_low_mid} neutron stars per century."
        )
        log.info(
            f"Galactic neutron star birth rate per century according to HTRU high survey: {birth_rate_HTRU_high} neutron stars per century."
        )

        # Add the information of the birth rates to the configuration file.
        cfg["birth_rate_PMPS"] = birth_rate_PMPS
        cfg["birth_rate_SMPS"] = birth_rate_SMPS
        cfg["birth_rate_HTRU_low_mid"] = birth_rate_HTRU_low_mid
        cfg["birth_rate_HTRU_high"] = birth_rate_HTRU_high

        # Add the parameters from the dynamical database to the configuration file.
        cfg["t_max"] = conf_json["t_age_max"]
        cfg["NS_number"] = conf_json["NS_number"]

        # Add the path of the dynamical database in the configuration file.
        cfg["dyn_database_path"] = args.dyn_data

        # Dump updated configuration to output path.
        config_dump_path = pathlib.Path().joinpath(
            output_path, "configuration.json"
        )
        with open(config_dump_path, "w") as f:
            json.dump(cfg, f, indent=4, sort_keys=True)

        # ===================== EXPORT OUTPUT ========================

        with timewith.TimeWith(
            "[Export]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Adding the parameters of the detected neutron stars to a data frame for export.
            log.info("Creating data frame for exporting...")

            # Generating two header lines and merging them using MultiIndex.
            parameters_final = [
                "age",
                "RA",
                "DEC",
                "l",
                "b",
                "DM",
                "d",
                "pm_RA",
                "pm_DEC",
                "v_ls",
                "B",
                "chi",
                "P",
                "P_dot",
                "L_radio_bol",
                "S_radio_obs_mean",
                "w_int",
                "w_eff",
            ]
            units_final = [
                "[yr]",
                "[deg]",
                "[deg]",
                "[deg]",
                "[deg]",
                "[pc cm^-3]",
                "[kpc]",
                "[mas yr^-1]",
                "[mas yr^-1]",
                "[km s^-1]",
                "[G]",
                "[rad]",
                "[s]",
                "[s s^-1]",
                "[erg s^-1]",
                "[Jy]",
                "[s]",
                "[s]",
            ]

            parameters_final_HTRU = [
                "age",
                "RA",
                "DEC",
                "l",
                "b",
                "DM",
                "d",
                "pm_RA",
                "pm_DEC",
                "v_ls",
                "B",
                "chi",
                "P",
                "P_dot",
                "L_radio_bol",
                "S_radio_obs_mean",
                "w_int",
                "w_eff",
                "HTRU_low",
                "HTRU_mid",
                "HTRU_high",
            ]
            units_final_HTRU = [
                "[yr]",
                "[deg]",
                "[deg]",
                "[deg]",
                "[deg]",
                "[pc cm^-3]",
                "[kpc]",
                "[mas yr^-1]",
                "[mas yr^-1]",
                "[km s^-1]",
                "[G]",
                "[rad]",
                "[s]",
                "[s s^-1]",
                "[erg s^-1]",
                "[Jy]",
                "[s]",
                "[s]",
                "",
                "",
                "",
            ]
            header_final = pd.MultiIndex.from_arrays(
                [parameters_final, units_final]
            )
            header_final_HTRU = pd.MultiIndex.from_arrays(
                [parameters_final_HTRU, units_final_HTRU]
            )
            df_PMPS = pd.DataFrame.from_dict(data=dictionary_detected_PMPS)

            df_PMPS.columns = header_final

            df_SMPS = pd.DataFrame.from_dict(data=dictionary_detected_SMPS)

            df_SMPS.columns = header_final

            df_HTRU_low = pd.DataFrame.from_dict(
                data=dictionary_detected_HTRU_low
            )

            df_HTRU_mid = pd.DataFrame.from_dict(
                data=dictionary_detected_HTRU_mid
            )

            df_HTRU_high = pd.DataFrame.from_dict(
                data=dictionary_detected_HTRU_high
            )

            df_HTRU = pd.concat([df_HTRU_low, df_HTRU_mid], axis=0)

            df_HTRU.columns = header_final_HTRU
            df_HTRU_high.columns = header_final_HTRU
            # Save the data frame as a compressed binary file.
            PMPS_output_path = pathlib.Path().joinpath(
                output_path, "survey_PMPS_results.pkl.gz"
            )
            df_PMPS.to_pickle(PMPS_output_path, compression="gzip")

            SMPS_output_path = pathlib.Path().joinpath(
                output_path, "survey_SMPS_results.pkl.gz"
            )
            df_SMPS.to_pickle(SMPS_output_path, compression="gzip")

            HTRU_output_low_mid_path = pathlib.Path().joinpath(
                output_path, "survey_HTRU_results_low_mid.pkl.gz"
            )

            df_HTRU.to_pickle(HTRU_output_low_mid_path, compression="gzip")

            HTRU_high_output_path = pathlib.Path().joinpath(
                output_path, "survey_HTRU_results_high.pkl.gz"
            )
            df_HTRU_high.to_pickle(HTRU_high_output_path, compression="gzip")

            log.info(
                f"Output of the detected population with PMPS generated in {os.getcwd()}/{PMPS_output_path}"
            )
            log.info(
                f"Output of the detected population with SMPS generated in {os.getcwd()}/{SMPS_output_path}"
            )
            log.info(
                f"Output of the detected population with HTRU low and mid surveys generated in {os.getcwd()}/{HTRU_output_low_mid_path}"
            )
            log.info(
                f"Output of the detected population with HTRU high surveys generated in {os.getcwd()}/{HTRU_high_output_path}"
            )

        # Cleanup. Reset seed to empty value.
        cfg["seed_magrot"] = None


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--dyn_data",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the file where the dynamically evolved population database is saved.",
    )

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
