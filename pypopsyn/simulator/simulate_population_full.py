"""
    Simulating a final population of neutron stars.

    An initial neutron star population of uniformly distributed ages is generated
    and the respective objects evolved in time according to their age.
    We simulate both the dynamical evolution in the Galaxy and the magneto-rotational
    evolution.
    Finally we model the radio emission and simulate the detection from two radio surveys,
    Parkes multibeam (PMPS) and Swinburne (SMPS).

    Display help message to run the code:

    python simulate_population_full.py --h

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


def simulate_population(args: argparse.Namespace) -> None:
    """
    Generating a neutron star population starting from some initial
    conditions and dynamically evolving it forward in time.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:
            save_dir (pathlib.Path): Output directory for the run.
            json_override_path (pathlib.Path): Path to JSON with parameter overrides.
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

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_full"] is None:
        cfg["seed_full"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_full"]))
    np.random.seed(cfg["seed_full"])

    # Update simulator configuration with the provided JSON override (if any).
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

    survey_PMPS = sr.SurveyRadio(PMPS_par_path)
    survey_SMPS = sr.SurveyRadio(SMPS_par_path)
    survey_HTRU_low = sr.SurveyRadio(HTRU_low_par_path)
    survey_HTRU_mid = sr.SurveyRadio(HTRU_mid_par_path)
    survey_HTRU_high = sr.SurveyRadio(HTRU_high_par_path)

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

            dist_cutoff = sun_dist_icrs < 35.0

            # Determine which stars fall into the sky region covered by any of the considered radio surveys.
            coverage_tot = (
                coverage_PMPS
                | coverage_SMPS
                | coverage_HTRU_low
                | coverage_HTRU_mid
                | coverage_HTRU_high
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
                S_radio_bol,
                intercepted_radio,
                coverage_PMPS,
                dist_cutoff,
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
                S_radio_bol,
                intercepted_radio,
                coverage_SMPS,
                dist_cutoff,
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
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_low,
                dist_cutoff,
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
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_mid,
                dist_cutoff,
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
                S_radio_bol,
                intercepted_radio,
                coverage_HTRU_high,
                dist_cutoff,
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

            NS_idx_PMPS = NS_idx[detected_radio_PMPS]
            NS_idx_SMPS = NS_idx[detected_radio_SMPS]
            NS_idx_HTRU_low = NS_idx[detected_radio_HTRU_low]
            NS_idx_HTRU_mid = NS_idx[detected_radio_HTRU_mid]
            NS_idx_HTRU_high = NS_idx[detected_radio_HTRU_high]

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
        "--parameter_override",
        nargs="?",
        type=str,
        default=None,
        help="Path to JSON containing the parameter override values.",
    )

    args = args.parse_args()

    simulate_population(args)
