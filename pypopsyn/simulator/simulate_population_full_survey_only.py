"""
    Simulating a range of surveys based on a simulation run performed with the full simulation scripts.

    Display help message to run the code:

    python simulate_population_full.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Vanessa Graber (Vanessa.Graber@rhul.ac.uk)
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

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.config_simulator as configuration
import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
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

            # Load a final_population.pkl.gz.
            NS_population_initial = ipop.InitialNeutronStarPopulation(
                cfg["NS_number"]
            )

            # Generate an array of indices.
            NS_idx = np.arange(cfg["NS_number"], dtype=int)

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
            # Convert from polar coordinates to Cartesian coordinates.
            x_initial, y_initial = coco.polar_to_cartesian(
                r_initial, phi_initial
            )

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

            timer.checkpoint("[Initial angular momentum]")

            # Computing the initial field strengths, misalignment angles, and periods
            log.info("Computing initial field strengths...")
            B_initial = NS_population_initial.magnetic_field()

            log.info("Computing initial misalignment angles...")
            chi_initial = NS_population_initial.misalignment_angle()

            log.info("Computing initial periods...")
            P_initial = NS_population_initial.period()

            timer.checkpoint(
                "[Initial field strengths, misalignment angles and periods]"
            )

            # Determining the initial period derivatives.
            log.info("Computing initial period derivatives...")
            period_derivative_vect = np.vectorize(pdv.period_derivative)
            P_dot_initial = (
                period_derivative_vect(B_initial, chi_initial, P_initial)
                / const.YR_TO_S
            )

            timer.checkpoint("[Initial period derivatives]")

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
                "[kpc yr^-1]",
                "[kpc yr^-1]",
                "[kpc yr^-1]",
                "[kpc yr^-1]",
                "[G]",
                "[rad]",
                "[s]",
                "[s s^-1]",
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

            timer.checkpoint("[Export]")

            log.info(
                f"Output of the initial population generated in {os.getcwd()}/{initial_output_path}"
            )

        # ===================== DYNAMICAL EVOLUTION ========================

        with timewith.TimeWith(
            "[DynamicalEvolution]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Evolve the initial population.
            log.info("Evolve the initial population in time dynamically...")

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
            dyn_evol_output, dyn_evol_dict = dyn.dynamical_evolution(
                initial_cond, age
            )

            r_final = dyn_evol_output[:, 0]
            phi_final = dyn_evol_output[:, 1]
            z_final = dyn_evol_output[:, 2]
            v_r_final = dyn_evol_output[:, 3]
            v_phi_final = dyn_evol_output[:, 4]
            v_z_final = dyn_evol_output[:, 5]

            # Convert from polar coordinates to Cartesian coordinates.
            x_final, y_final = coco.polar_to_cartesian(r_final, phi_final)

            # Convert velocities from [kpc/yr] into [km/s].
            v_r_final = v_r_final * const.KPC_TO_KM / const.YR_TO_S
            v_phi_final = v_phi_final * const.KPC_TO_KM / const.YR_TO_S
            v_z_final = v_z_final * const.KPC_TO_KM / const.YR_TO_S

            # Convert velocity component from galactocentric cylindrical coordinates
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
            timer.checkpoint("[Dynamical evolution]")

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

        # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================

        with timewith.TimeWith(
            "[MagnetoRotationalEvolution]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Determine the evolved magnetic field, misalignment angle and rotation period.
            log.info(
                "Evolving magnetic field, misalignment angle and rotation period..."
            )
            (
                B_final,
                chi_final,
                P_final,
                magrot_evol_dict,
            ) = mre.magneto_rotational_evolution(
                B_initial, chi_initial, P_initial, age, cfg["a_late"]
            )

            if cfg["save_magrot_evolution"]:
                # Save dictionary containing evolution information to output path in a .json file.
                magrot_evolution_dump_path = pathlib.Path().joinpath(
                    output_path, "magrot_evolution.json"
                )

                with open(magrot_evolution_dump_path, "wb") as f:
                    f.write(
                        orjson.dumps(
                            dict(magrot_evol_dict),
                            option=orjson.OPT_SERIALIZE_NUMPY
                            | orjson.OPT_NON_STR_KEYS
                            | orjson.OPT_SORT_KEYS,
                        )
                    )

            timer.checkpoint(
                "[Final field strengths, misalignment angles and periods]"
            )

            # Determining the final period derivatives.
            log.info("Computing final period derivatives...")
            P_dot_final = (
                period_derivative_vect(
                    B_final,
                    chi_final,
                    P_final,
                )
                / const.YR_TO_S
            )

            timer.checkpoint("[Final period derivatives]")

        # ===================== RADIO EMISSION ========================

        with timewith.TimeWith(
            "[RadioEmission]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ) as timer:

            # Calculate the relevant radio emission parameters.
            (
                intercepted_radio,
                S_radio_bol,
                w_int_s,
                L_radio_bol,
            ) = er.calculate_radio_emission_full(
                P_final, P_dot_final, sun_dist_icrs, chi_final
            )

            # Determine fraction of pulsars beamed towards us.
            fraction_intercepted = len(
                intercepted_radio[intercepted_radio]
            ) / len(intercepted_radio)

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
                ra_final, dec_final, l_final, b_final
            )
            coverage_SMPS = survey_SMPS.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_HTRU_low = survey_HTRU_low.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_HTRU_mid = survey_HTRU_mid.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_HTRU_high = survey_HTRU_high.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_SKA_low_AAstar = survey_SKA_low_AAstar.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_SKA_low_AA4 = survey_SKA_low_AA4.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_SKA_mid_band1_AAstar = (
                survey_SKA_mid_band1_AAstar.sky_coverage(
                    ra_final, dec_final, l_final, b_final
                )
            )
            coverage_SKA_mid_band2_AAstar = (
                survey_SKA_mid_band2_AAstar.sky_coverage(
                    ra_final, dec_final, l_final, b_final
                )
            )
            coverage_SKA_mid_band1_AA4 = survey_SKA_mid_band1_AA4.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )
            coverage_SKA_mid_band2_AA4 = survey_SKA_mid_band2_AA4.sky_coverage(
                ra_final, dec_final, l_final, b_final
            )

            dist_cutoff = sun_dist_icrs < 35.0

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

            fraction_coverage = (
                np.count_nonzero(coverage_tot) / cfg["NS_number"]
            )
            log.info(
                f"Fraction of pulsars in the covered sky region: {fraction_coverage}"
            )

            timer.checkpoint("[Total sky coverage]")

            # Determine which stars could in principle be detected.
            detectable_radio = intercepted_radio & coverage_tot

            # Computing the DM for the stars that fall into the surveys' sky coverage and whose
            # radio beam intercepts our line of sight.
            DM = np.zeros(cfg["NS_number"])
            DM[detectable_radio] = edm.compute_DM(
                l_final[detectable_radio],
                b_final[detectable_radio],
                sun_dist_gal[detectable_radio],
                cfg["ed_model"],
            )

            timer.checkpoint("[DM computation]")

            # Computing the spectral index and scattering timescale at 327 MHz of each star.
            spectral_index = np.random.normal(
                cfg["mean_spectral_index"],
                cfg["std_spectral_index"],
                len(S_radio_bol),
            )
            tau_sc = np.zeros(cfg["NS_number"])
            tau_sc[DM != 0] = edm.compute_tau_sc_327(DM[DM != 0])

            # Simulating the PMPS survey.
            log.info("Simulate detection with PMPS...")

            (
                detected_radio_PMPS,
                S_radio_obs_mean_PMPS,
                w_eff_PMPS,
                S_radio_obs_PMPS,
            ) = survey_PMPS.detected_radio_population_full(
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_PMPS,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SMPS,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_low,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_mid,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_high,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_low_AAstar,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_low_AA4,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_mid_band1_AAstar,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_mid_band2_AAstar,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_mid_band1_AA4,
                dist_cutoff,
                spectral_index,
                tau_sc,
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
                w_int_s,
                DM,
                P_final,
                l_final,
                b_final,
                dec_final,
                S_radio_bol,
                intercepted_radio,
                coverage_SKA_mid_band2_AA4,
                dist_cutoff,
                spectral_index,
                tau_sc,
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

        # Exporting the final population file containing the intrinsic properties of the entire population.

        # Generating two header lines and merging them using MultiIndex.
        parameters_final = [
            "age",
            "x",
            "y",
            "z",
            "RA",
            "DEC",
            "l",
            "b",
            "d",
            "v_r",
            "v_phi",
            "v_z",
            "pm_RA",
            "pm_DEC",
            "v_ls",
            "B",
            "chi",
            "P",
            "P_dot",
            "L_radio_bol",
            "S_radio_bol",
            "w_int",
            "intercepted_radio",
            "spectral_index",
        ]
        units_final = [
            "[yr]",
            "[kpc]",
            "[kpc]",
            "[kpc]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[kpc]",
            "[km s^-1]",
            "[km s^-1]",
            "[km s^-1]",
            "[mas yr^-1]",
            "[mas yr^-1]",
            "[km s^-1]",
            "[G]",
            "[rad]",
            "[s]",
            "[s s^-1]",
            "[erg s^-1]",
            "[erg s^-1 cm^(-2)]",
            "[s]",
            " ",
            " ",
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
                    l_final,
                    b_final,
                    sun_dist_icrs,
                    v_r_final,
                    v_phi_final,
                    v_z_final,
                    pm_ra_final,
                    pm_dec_final,
                    v_ls_icrs,
                    B_final,
                    chi_final,
                    P_final,
                    P_dot_final,
                    L_radio_bol,
                    S_radio_bol,
                    w_int_s,
                    intercepted_radio,
                    spectral_index,
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
                    ra_final[NS_idx_PMPS],
                    dec_final[NS_idx_PMPS],
                    l_final[NS_idx_PMPS],
                    b_final[NS_idx_PMPS],
                    sun_dist_icrs[NS_idx_PMPS],
                    DM[NS_idx_PMPS],
                    pm_ra_final[NS_idx_PMPS],
                    pm_dec_final[NS_idx_PMPS],
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
                    ra_final[NS_idx_SMPS],
                    dec_final[NS_idx_SMPS],
                    l_final[NS_idx_SMPS],
                    b_final[NS_idx_SMPS],
                    sun_dist_icrs[NS_idx_SMPS],
                    DM[NS_idx_SMPS],
                    pm_ra_final[NS_idx_SMPS],
                    pm_dec_final[NS_idx_SMPS],
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
                    ra_final[NS_idx_HTRU_high],
                    dec_final[NS_idx_HTRU_high],
                    l_final[NS_idx_HTRU_high],
                    b_final[NS_idx_HTRU_high],
                    sun_dist_icrs[NS_idx_HTRU_high],
                    DM[NS_idx_HTRU_high],
                    pm_ra_final[NS_idx_HTRU_high],
                    pm_dec_final[NS_idx_HTRU_high],
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
                    ra_final[NS_idx_HTRU_low],
                    dec_final[NS_idx_HTRU_low],
                    l_final[NS_idx_HTRU_low],
                    b_final[NS_idx_HTRU_low],
                    sun_dist_icrs[NS_idx_HTRU_low],
                    DM[NS_idx_HTRU_low],
                    pm_ra_final[NS_idx_HTRU_low],
                    pm_dec_final[NS_idx_HTRU_low],
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
                    ra_final[NS_idx_HTRU_mid],
                    dec_final[NS_idx_HTRU_mid],
                    l_final[NS_idx_HTRU_mid],
                    b_final[NS_idx_HTRU_mid],
                    sun_dist_icrs[NS_idx_HTRU_mid],
                    DM[NS_idx_HTRU_mid],
                    pm_ra_final[NS_idx_HTRU_mid],
                    pm_dec_final[NS_idx_HTRU_mid],
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
                    ra_final[NS_idx_SKA_low_AAstar],
                    dec_final[NS_idx_SKA_low_AAstar],
                    l_final[NS_idx_SKA_low_AAstar],
                    b_final[NS_idx_SKA_low_AAstar],
                    sun_dist_icrs[NS_idx_SKA_low_AAstar],
                    DM[NS_idx_SKA_low_AAstar],
                    pm_ra_final[NS_idx_SKA_low_AAstar],
                    pm_dec_final[NS_idx_SKA_low_AAstar],
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
                    ra_final[NS_idx_SKA_low_AA4],
                    dec_final[NS_idx_SKA_low_AA4],
                    l_final[NS_idx_SKA_low_AA4],
                    b_final[NS_idx_SKA_low_AA4],
                    sun_dist_icrs[NS_idx_SKA_low_AA4],
                    DM[NS_idx_SKA_low_AA4],
                    pm_ra_final[NS_idx_SKA_low_AA4],
                    pm_dec_final[NS_idx_SKA_low_AA4],
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
                    ra_final[NS_idx_SKA_mid_band1_AAstar],
                    dec_final[NS_idx_SKA_mid_band1_AAstar],
                    l_final[NS_idx_SKA_mid_band1_AAstar],
                    b_final[NS_idx_SKA_mid_band1_AAstar],
                    sun_dist_icrs[NS_idx_SKA_mid_band1_AAstar],
                    DM[NS_idx_SKA_mid_band1_AAstar],
                    pm_ra_final[NS_idx_SKA_mid_band1_AAstar],
                    pm_dec_final[NS_idx_SKA_mid_band1_AAstar],
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
                    ra_final[NS_idx_SKA_mid_band2_AAstar],
                    dec_final[NS_idx_SKA_mid_band2_AAstar],
                    l_final[NS_idx_SKA_mid_band2_AAstar],
                    b_final[NS_idx_SKA_mid_band2_AAstar],
                    sun_dist_icrs[NS_idx_SKA_mid_band2_AAstar],
                    DM[NS_idx_SKA_mid_band2_AAstar],
                    pm_ra_final[NS_idx_SKA_mid_band2_AAstar],
                    pm_dec_final[NS_idx_SKA_mid_band2_AAstar],
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
                    ra_final[NS_idx_SKA_mid_band1_AA4],
                    dec_final[NS_idx_SKA_mid_band1_AA4],
                    l_final[NS_idx_SKA_mid_band1_AA4],
                    b_final[NS_idx_SKA_mid_band1_AA4],
                    sun_dist_icrs[NS_idx_SKA_mid_band1_AA4],
                    DM[NS_idx_SKA_mid_band1_AA4],
                    pm_ra_final[NS_idx_SKA_mid_band1_AA4],
                    pm_dec_final[NS_idx_SKA_mid_band1_AA4],
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
                    ra_final[NS_idx_SKA_mid_band2_AA4],
                    dec_final[NS_idx_SKA_mid_band2_AA4],
                    l_final[NS_idx_SKA_mid_band2_AA4],
                    b_final[NS_idx_SKA_mid_band2_AA4],
                    sun_dist_icrs[NS_idx_SKA_mid_band2_AA4],
                    DM[NS_idx_SKA_mid_band2_AA4],
                    pm_ra_final[NS_idx_SKA_mid_band2_AA4],
                    pm_dec_final[NS_idx_SKA_mid_band2_AA4],
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
        default="output/sim_full",
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
