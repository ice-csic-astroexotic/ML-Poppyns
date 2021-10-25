"""
Simulating a detected population of neutron stars from a dynamically evolved population database.

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
import random
import sys
import time

import numpy as np
import pandas as pd

import pypopsyn.benchmark.timewith as timewith
import pypopsyn.simulator.basics.cdf_calculator as cc
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population as ipop
import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


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

    # Check if the parsed dynamically simulated population directory exists.
    dyn_path = pathlib.Path(args.dyn_data)
    dyn_path = pathlib.Path().joinpath(dyn_path, "final_pop_dyn.pkl.gz")
    dyn_path_config = pathlib.Path().joinpath(dyn_path, "override.json")
    if not dyn_path.exists():
        log.error(f"File {dyn_path} not found...")
        sys.exit()

    # Update path-dependent configurations prepending the specified output path.
    configuration.cfg["profile_log"] = str(
        pathlib.Path().joinpath(output_path, configuration.cfg["profile_log"])
    )
    configuration.cfg["profile_json"] = str(
        pathlib.Path().joinpath(output_path, configuration.cfg["profile_json"])
    )

    # Initialize seed randomly if no seed was specified.
    if cfg["seed_magrot"] is None:
        cfg["seed_magrot"] = int(time.time())

    # Set NumPy random seed globally.
    log.info("Seed: {}".format(cfg["seed_magrot"]))
    np.random.seed(cfg["seed_magrot"])

    # Update simulator configuration with the dynamical database JSON override (if any).
    cfg_override_dyn = {}
    if dyn_path_config.exists():
        json_override_path = dyn_path_config
        with open(json_override_path) as f:
            cfg_override_dyn = json.load(f)
            configuration.update_configuration(cfg_override_dyn)

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
        json.dump(configuration.cfg, f, indent=4, sort_keys=True)

    # Initialize the surveys.
    survey_PMPS = sr.SurveyRadioPMPS()
    survey_SMPS = sr.SurveyRadioSMPS()

    n_detected_real_PMPS = cfg["detected_real_PMPS"]
    n_detected_real_SMPS = cfg["detected_real_SMPS"]

    with timewith.TimeWith(
        "[TotalSimulation]",
        cfg["profile_log"],
        cfg["profile_json"],
        cfg["show_profiling"],
    ):
        with timewith.TimeWith(
            "[LoadPopulationDynamics]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):

            # Load the file containing the dynamically evolved population parameters.
            df_dyn = pd.read_pickle(f"{dyn_path}", compression="gzip")

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

            detected_dictionary = {
                "age": [],
                "x": [],
                "y": [],
                "z": [],
                "ra": [],
                "dec": [],
                "l": [],
                "b": [],
                "DM": [],
                "dist": [],
                "v_r": [],
                "v_phi": [],
                "v_z": [],
                "pm_ra": [],
                "pm_dec": [],
                "v_ls": [],
                "B": [],
                "chi": [],
                "P": [],
                "Pdot": [],
                "L_radio": [],
                "S_radio": [],
                "w_int": [],
                "PMPS": [],
                "SMPS": [],
            }

            n_created = 0
            n_created_PMPS = 0
            n_created_SMPS = 0
            n_detected_sim_PMPS = 0
            n_detected_sim_SMPS = 0
            stop_PMPS = False
            stop_SMPS = False

            # To speed up the simulation, generate new neutron stars in batches.
            n_batchsize = 100000

            flag_80 = False
            flag_95 = False

            # Continue to simulate stars until the detected number of pulsars for all the surveys is reached.
            while (n_detected_sim_PMPS < n_detected_real_PMPS) | (
                n_detected_sim_SMPS < n_detected_real_SMPS
            ):

                # Evaluate the percentage of neutron stars detected by the simulated
                # surveys with respect to the real surveys.
                percentage_detected_PMPS = (
                    n_detected_sim_PMPS / n_detected_real_PMPS
                )
                percentage_detected_SMPS = (
                    n_detected_sim_SMPS / n_detected_real_SMPS
                )

                # If the percentage of both surveys is over 80% reduce the batch size.
                if (
                    (percentage_detected_PMPS > 0.8)
                    & (percentage_detected_SMPS > 0.8)
                    & (flag_80 is False)
                ):
                    flag_80 = True
                    n_batchsize = 10000
                elif (
                    (percentage_detected_PMPS > 0.95)
                    & (percentage_detected_SMPS > 0.95)
                    & (flag_95 is False)
                ):
                    flag_95 = True
                    n_batchsize = 5000

                # Update the total number of simulated neutron stars.
                n_created += n_batchsize

                log.info(f"Total number of created neutron stars: {n_created}")

                # Select random neutron stars from the database.
                idx = np.array(
                    random.sample(range(len(ra_final)), n_batchsize)
                )

                age_d = np.array(age[idx])
                ra_d = np.array(ra_final[idx])
                dec_d = np.array(dec_final[idx])
                l_d = np.array(l_final[idx])
                b_d = np.array(b_final[idx])
                dist_d = np.array(sun_dist_icrs[idx])

                # Select only neutron stars that fall in the sky region covered by the surveys.
                coverage_PMPS = survey_PMPS.sky_coverage(ra_d, dec_d, l_d, b_d)
                coverage_SMPS = survey_SMPS.sky_coverage(ra_d, dec_d, l_d, b_d)

                coverage_tot = coverage_PMPS | coverage_SMPS
                idx_det = idx[coverage_tot]

                age_d = age_d[coverage_tot]
                l_d = l_d[coverage_tot]
                b_d = b_d[coverage_tot]
                dist_d = dist_d[coverage_tot]
                coverage_PMPS = coverage_PMPS[coverage_tot]
                coverage_SMPS = coverage_SMPS[coverage_tot]

                # Remove stars that fall out from the total sky coverage.
                out_coverage = np.invert(coverage_tot)
                idx_remove = idx[out_coverage]

                # Initialize neutron star population properties.
                pop_initial = ipop.InitialNeutronStarPopulation(
                    NS_number=len(age_d)
                )

                # Computing the initial field strengths, misalignment angles, and periods.
                B_initial = pop_initial.magnetic_field()
                chi_initial = pop_initial.misalignment_angle()
                P_initial = pop_initial.period()

                # Determine the evolved magnetic field, misalignment angle and rotation period.
                (
                    B_d,
                    chi_d,
                    P_d,
                    NS_magrot_evol_dict,
                ) = mre.magneto_rotational_evolution(
                    B_initial, chi_initial, P_initial, age_d,
                )

                # Determining the radio beam angular aperture.
                theta_beam = er.beam_aperture(P_d, cfg["r_em"])

                # Determining the fraction of solid angle spanned by the two radio beams in a star complete rotation.
                beam_frac = er.beam_fraction(chi_d, theta_beam)

                # Drawing a random angular intercept for the line of sight.
                # Note that since we assume symmetry between the northern and southern hemisphere of the star
                # we only need to consider one hemisphere, e.g., the northern one.
                los_grid = np.linspace(0.0, np.pi / 2, cfg["resolution"])
                los_rand = cc.random_from_pdf(los_grid, np.sin, len(age_d))

                # Determining if the pulsar's radio beam intercepts our line of sight.
                intercepted_radio = er.los_intercept(
                    chi_d, theta_beam, los_rand,
                )

                # Select only neutron stars that points at us.
                idx_det = idx_det[intercepted_radio]

                age_d = age_d[intercepted_radio]
                l_d = l_d[intercepted_radio]
                b_d = b_d[intercepted_radio]
                dist_d = dist_d[intercepted_radio]
                B_d = B_d[intercepted_radio]
                chi_d = chi_d[intercepted_radio]
                P_d = P_d[intercepted_radio]
                theta_beam_d = theta_beam[intercepted_radio]
                los_rand_d = los_rand[intercepted_radio]
                beam_frac_d = beam_frac[intercepted_radio]
                coverage_PMPS = coverage_PMPS[intercepted_radio]
                coverage_SMPS = coverage_SMPS[intercepted_radio]

                if np.count_nonzero(intercepted_radio) == 0:
                    break

                # Determining the final period derivative.
                period_derivative_vect = np.vectorize(pdv.period_derivative)
                P_dot_d = period_derivative_vect(B_d, chi_d, P_d,)

                # Determining the luminosity in different electromagnetic bands.
                L_radio = er.pdf_radio_luminosity(
                    P_d, P_dot_d / const.YR_TO_S,
                )

                # Computing the intrinsic pulse width of the radio pulse.
                w_intrinsic = er.pulse_width(chi_d, theta_beam_d, los_rand_d,)
                # Convert pulse width from [rad] to [s].
                w_intrinsic_s = w_intrinsic * P_d / (2.0 * np.pi)

                # Computing the radio flux observed on Earth.
                S_radio = er.erg_flux_radio(
                    L_radio, dist_d, beam_frac_d, w_intrinsic,
                )
                # Convert Radio flux in Jy.
                S_radio_Jy = S_radio / const.JY_TO_ERG

                # Computing the DM.
                DM = edm.compute_DM(l_d, b_d, dist_d, cfg["ed_model"],)

                # Simulating the PMPS survey.
                detected_radio_PMPS = np.zeros(len(age_d), dtype=bool)

                detected_radio_PMPS[coverage_PMPS] = survey_PMPS.detect(
                    S_radio_Jy[coverage_PMPS],
                    DM[coverage_PMPS],
                    l_d[coverage_PMPS],
                    b_d[coverage_PMPS],
                    w_intrinsic_s[coverage_PMPS],
                    P_d[coverage_PMPS],
                )

                n_detected_sim_PMPS += np.count_nonzero(detected_radio_PMPS)
                log.info(
                    f"Total number of neutron stars detected by the Parkes multibeam survey: {n_detected_sim_PMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with PMPS is reached.
                # This is needed to compute the birth rate derived from the PMPS detections.
                if (n_detected_sim_PMPS > n_detected_real_PMPS) & (
                    stop_PMPS is False
                ):
                    stop_PMPS = True
                    n_created_PMPS = n_created

                # simulating the SMPS survey.
                detected_radio_SMPS = np.zeros(len(age_d), dtype=bool)

                detected_radio_SMPS[coverage_SMPS] = survey_SMPS.detect(
                    S_radio_Jy[coverage_SMPS],
                    DM[coverage_SMPS],
                    l_d[coverage_SMPS],
                    b_d[coverage_SMPS],
                    w_intrinsic_s[coverage_SMPS],
                    P_d[coverage_SMPS],
                )

                n_detected_sim_SMPS += np.count_nonzero(detected_radio_SMPS)
                log.info(
                    f"Total number of neutron stars detected by the Swinburne pulsar survey: {n_detected_sim_SMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with SMPS is reached.
                # This is needed to compute the birth rate derived from the SMPS detections.
                if (n_detected_sim_SMPS > n_detected_real_SMPS) & (
                    stop_SMPS is False
                ):
                    stop_SMPS = True
                    n_created_SMPS = n_created

                # Select only neutron stars that are detected by one of the surveys.
                detected = detected_radio_PMPS | detected_radio_SMPS
                idx_det = idx_det[detected]

                # Update the database of detected neutron stars.
                update_detected_dictionary = {
                    "age": age[idx_det].tolist(),
                    "x": age[idx_det].tolist(),
                    "y": y_final[idx_det].tolist(),
                    "z": z_final[idx_det].tolist(),
                    "ra": ra_final[idx_det].tolist(),
                    "dec": dec_final[idx_det].tolist(),
                    "l": l_final[idx_det].tolist(),
                    "b": b_final[idx_det].tolist(),
                    "DM": DM[detected].tolist(),
                    "dist": sun_dist_icrs[idx_det].tolist(),
                    "v_r": v_r_final[idx_det].tolist(),
                    "v_phi": v_phi_final[idx_det].tolist(),
                    "v_z": v_z_final[idx_det].tolist(),
                    "pm_ra": pm_ra_final[idx_det].tolist(),
                    "pm_dec": pm_dec_final[idx_det].tolist(),
                    "v_ls": v_ls_icrs[idx_det].tolist(),
                    "B": B_d[detected].tolist(),
                    "chi": chi_d[detected].tolist(),
                    "P": P_d[detected].tolist(),
                    "Pdot": P_dot_d[detected].tolist(),
                    "L_radio": L_radio[detected].tolist(),
                    "S_radio": S_radio_Jy[detected].tolist(),
                    "w_int": w_intrinsic_s[detected].tolist(),
                    "PMPS": detected_radio_PMPS[detected].tolist(),
                    "SMPS": detected_radio_SMPS[detected].tolist(),
                }

                # Update the dictionary containing the detection information.
                detected_dictionary = {
                    key: value + update_detected_dictionary[key]
                    for key, value in detected_dictionary.items()
                }

                # Remove from the dynamical database the stars that have been detected or
                # that are out from the sky coverage of the surveys.
                idx_remove = np.concatenate((idx_remove, idx_det), axis=None)

                age = np.delete(age, idx_remove)
                ra_final = np.delete(ra_final, idx_remove)
                dec_final = np.delete(dec_final, idx_remove)
                l_final = np.delete(l_final, idx_remove)
                b_final = np.delete(b_final, idx_remove)
                sun_dist_icrs = np.delete(sun_dist_icrs, idx_remove)

            # Determine the Galactic neutron star birth rate for the different surveys.
            t_max = cfg["t_age_max"] / 100  # Maximum time in centuries.
            log.info(
                f"Galactic neutron star birth rate according to PMPS: {n_created_PMPS/t_max} neutron stars per century."
            )
            log.info(
                f"Galactic neutron star birth rate according to SMPS: {n_created_SMPS / t_max} neutron stars per century."
            )

        ###################################################################################################

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
                "x",
                "y",
                "z",
                "RA",
                "DEC",
                "l",
                "b",
                "DM",
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
                "L_radio",
                "S_radio",
                "w_int",
                "survey_PMPS",
                "survey_SMPS",
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
                "[pc cm^-3]",
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
                "[s yr^-1]",
                "[erg s^-1 Hz^-1]",
                "[Jy]",
                "[s]",
                " ",
                " ",
            ]
            header_final = pd.MultiIndex.from_arrays(
                [parameters_final, units_final]
            )

            df_final = pd.DataFrame.from_dict(data=detected_dictionary)
            df_final.columns = header_final

            # Save the data frame as a compressed binary file.
            final_output_path = pathlib.Path().joinpath(
                output_path, "final_population.pkl.gz"
            )
            df_final.to_pickle(final_output_path, compression="gzip")

            log.info(
                f"Output of the detected population generated in {os.getcwd()}/{final_output_path}"
            )

        # Cleanup. Reset seed to empty value.
        configuration.cfg["seed"] = None


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
