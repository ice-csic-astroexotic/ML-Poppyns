"""
    Simulating a detected population of neutron stars from a dynamically evolved population database.

    Display help message to run the code:

    python simulate_population_magrot_det.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import argparse
import functools
import json
import logging
import os
import pathlib
import pickle
import sys
import time
from typing import Tuple

import numpy as np
import orjson
import pandas as pd
from scipy.interpolate import RectBivariateSpline

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.config_simulator as configuration
import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_emission.emission_xray as ex
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.multiband_surveys.survey_x as sx
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.benchmark.timewith as timewith
import utilities.samplers.memory_efficient_sampling as mes
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def initialize_radio_surveys() -> Tuple[dict, dict]:
    """
    Initialize and return radio survey objects and empty detection dictionaries.

    Returns:
        (Tuple[dict, dict, dict, dict, dict]): A tuple of dictionaries containing the following information:

            - A dictionary of initialized radio survey objects, keyed by survey names (e.g., "PMPS", "SMPS", etc.).
            - An empty dictionary for storing detected neutron star data for each of the survey.
    """
    # Get the path to the software directory.
    base_path = pathlib.Path(cfg["path_to_software"])

    # Initialize survey objects.
    radio_surveys = {}

    for survey_name, survey_dict in cfg["surveys_radio"].items():
        if survey_name == "HTRU_low_mid":
            radio_surveys["HTRU_low"] = sr.SurveyRadio(
                str(base_path.joinpath(survey_dict["path_low"]))
            )
            radio_surveys["HTRU_mid"] = sr.SurveyRadio(
                str(base_path.joinpath(survey_dict["path_mid"]))
            )
        else:
            radio_surveys[survey_name] = sr.SurveyRadio(
                str(base_path.joinpath(survey_dict["path"]))
            )

    # Detection dictionary template.
    detection_template = {
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
        "P_dot": [],
        "L_radio_bol": [],
        "S_radio_obs_mean": [],
        "S_radio_obs_mean_1400": [],
        "w_int": [],
        "w_eff": [],
        "spectral_index": [],
        "idx": [],
    }

    # Initialize detection dictionaries for each survey.
    detection_dictionaries = {
        survey_name: detection_template.copy()
        for survey_name in cfg["surveys_radio"].keys()
    }

    # Add flags for HTRU low and mid surveys.
    if "HTRU_low_mid" in cfg["surveys_radio"]:
        detection_dictionaries["HTRU_low_mid"].update(
            {"HTRU_low": [], "HTRU_mid": []}
        )

    return radio_surveys, detection_dictionaries


def initialize_x_surveys() -> Tuple[dict, RectBivariateSpline]:
    """
    Initialize and return an x-ray survey detection dictionary and an interpolator for x-ray luminosity.

    Returns:
        (Tuple[dict, RectBivariateSpline]):
            - A dictionary for storing detected neutron star data for the x-ray survey.
            - An interpolator function loaded from a pickled file to evaluate the x-ray luminosity.
    """
    dictionary_detected_x = {
        "age": [],
        "ra": [],
        "dec": [],
        "l": [],
        "b": [],
        "N_H": [],
        "dist": [],
        "pm_ra": [],
        "pm_dec": [],
        "v_ls": [],
        "B": [],
        "chi": [],
        "P": [],
        "P_dot": [],
        "L_x_therm": [],
        "S_x_abs": [],
        "idx": [],
    }

    # Load the interpolator function to evaluate the x-ray luminosity.
    interpolator_Lx_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/interpolator_Lx.pkl",
    )
    with open(interpolator_Lx_path, "rb") as f:
        L_x_interpolator = pickle.load(f)

    return dictionary_detected_x, L_x_interpolator


def load_database_dyn(
    dyn_path: pathlib.Path,
    n_batchsize: int,
    idx_remove: list,
) -> dict:
    """
    Load a batch of a dynamically evolved population from a .csv file, convert coordinates and return a dictionary
    with the dynamical information of the selected stars.

    Args:
        dyn_path (pathlib.Path): Path to the directory containing the dynamically evolved population data.
        n_batchsize (int): Batch size of stars to select when loading the data.
        idx_remove (list): List of indices to remove from the dynamical database.

    Returns:
        (dict): A dictionary containing the data of the selected dynamical population chunk.
    """

    # Check if the parsed dynamically simulated population directory exists.
    dyn_path = pathlib.Path(dyn_path)
    dyn_config_path = dyn_path / "configuration.json"
    dyn_data_path = dyn_path / "final_pop_dyn.csv"

    if not dyn_data_path.exists():
        log.error(f"File {dyn_data_path} not found...")
        sys.exit()

    with open(dyn_config_path, "r") as f:
        config_dyn = json.load(f)

    # Load the batch of the file containing the dynamically evolved population parameters.
    df_dyn = mes.select(
        dyn_data_path,
        n_batchsize,
        config_dyn["NS_number"],
        idx_remove,
    )

    age = df_dyn["age"]["[yr]"].to_numpy()
    r = df_dyn["r"]["[kpc]"].to_numpy()
    phi = df_dyn["phi"]["[rad]"].to_numpy()
    z = df_dyn["z"]["[kpc]"].to_numpy()
    v_r = df_dyn["v_r"]["[km/s]"].to_numpy()
    v_phi = df_dyn["v_phi"]["[km/s]"].to_numpy()
    v_z = df_dyn["v_z"]["[km/s]"].to_numpy()

    # Convert from polar coordinates to Cartesian coordinates.
    x, y = coco.polar_to_cartesian(r, phi)

    # Convert velocity components from galactocentric cylindrical coordinates
    # to galactocentric Cartesian coordinates.
    (
        v_x,
        v_y,
        v_z,
    ) = coco.speed_cylindrical_to_cartesian(v_r, v_phi, v_z, phi)

    # Convert galactocentric coordinates and velocities into ICRS frame.
    (
        ra,
        dec,
        dist_heliocentric_icrs,
        pm_ra,
        pm_dec,
        v_ls_icrs,
    ) = coco.galactocentric_to_icrs(x, y, z, v_x, v_y, v_z)

    # Convert galactocentric coordinates and velocities into galactic coordinates.
    (
        l_gal,
        b_gal,
        dist_heliocentric_gal,
        pm_l,
        pm_b,
        v_ls_gal,
    ) = coco.galactocentric_to_galactic(x, y, z, v_x, v_y, v_z)

    dictionary_dyn_database_chunk = {
        "age": age,
        "ra": ra,
        "dec": dec,
        "l": l_gal,
        "b": b_gal,
        "dist": dist_heliocentric_icrs,
        "pm_ra": pm_ra,
        "pm_dec": pm_dec,
        "v_ls": v_ls_icrs,
        "idx": df_dyn.index.values,
    }

    # Update the parameters in the simulation configuration file with the ones of the dynamical database.
    cfg["t_age_max"] = config_dyn["t_age_max"]
    cfg["NS_number"] = config_dyn["NS_number"]
    cfg["kick_model"] = config_dyn["kick_model"]
    cfg["sigma_k"] = config_dyn["sigma_k"]
    cfg["vk_c"] = config_dyn["vk_c"]
    cfg["h_c"] = config_dyn["h_c"]

    # Add the path of the dynamical database in the configuration file.
    cfg["dyn_database_path"] = str(dyn_path)

    return dictionary_dyn_database_chunk


def apply_surveys_coverage(
    surveys_radio: dict,
    survey_xray: bool,
    dyn_database_dict: dict,
    idx_remove: list,
    dist_cutoff: float,
) -> Tuple[dict, list]:
    """
    Apply survey coverage criteria to filter a dynamic population dataset based on sky coverage,
    distance, and age cutoffs, and update the indices of entries to be removed.

    Args:
        surveys_radio (dict): A dictionary of radio survey objects, containing the information on the sky coverage.
        survey_xray (bool): A boolean flag to consider an all sky coverage for an X-ray survey.
        dyn_database_dict (dict): A dictionary containing the data of a dynamical population.
        idx_remove (list): A list of indices of entries to be removed based on the filtering criteria.
        dist_cutoff (float): The maximum heliocentric distance to include in the survey coverage.

    Returns:
        (Tuple[dict, list]): A tuple object containing the following attributes:

            - A dictionary containing data for stars that meet the coverage criteria.
            - An updated list of indices of stars that are outside the coverage and should be removed.
    """

    dist = dyn_database_dict["dist"]
    dist_mask = dist < dist_cutoff

    # Evaluate the sky coverage for each radio survey.
    survey_radio_names = list(surveys_radio.keys())
    coverage_survey_radio = {}

    for name in survey_radio_names:
        coverage_survey_radio[name] = surveys_radio[name].sky_coverage(
            dyn_database_dict["ra"],
            dyn_database_dict["dec"],
            dyn_database_dict["l"],
            dyn_database_dict["b"],
        )

    coverage_radio_tot = (
        functools.reduce(
            lambda a, b: a | b,
            (coverage_survey_radio[name] for name in survey_radio_names),
        )
    ) & dist_mask

    # For the x-ray survey we consider an all-sky coverage and only apply a distance cutoff.
    coverage_x_tot = dist_mask

    # Evaluate the total sky coverage for both radio and X-ray surveys.
    if survey_xray:
        coverage_tot = coverage_radio_tot | coverage_x_tot

    else:
        coverage_tot = coverage_radio_tot

    # Select only neutron stars that fall into the sky region covered by the surveys.
    dictionary_coverage_database = {
        key: value[coverage_tot] for key, value in dyn_database_dict.items()
    }

    # Add coverage for each survey to the dictionary.
    dictionary_coverage_database["coverage_radio"] = coverage_radio_tot[
        coverage_tot
    ]

    for survey_name in survey_radio_names:
        # Add the coverage data for each survey
        coverage_key = f"coverage_radio_{survey_name}"
        dictionary_coverage_database[coverage_key] = coverage_survey_radio[
            survey_name
        ][coverage_tot]

    if survey_xray:
        dictionary_coverage_database["coverage_x"] = coverage_x_tot[
            coverage_tot
        ]

    # Remove stars that do not fall into the total sky coverage.
    idx = dyn_database_dict["idx"]
    out_coverage = np.invert(coverage_tot)
    idx_remove += idx[out_coverage].tolist()

    return dictionary_coverage_database, idx_remove


def initialize_population_magrot(dict_coverage_database: dict) -> dict:
    """
    Initialize the magneto-rotational properties of a neutron star population falling in the survey sky coverage.

    Args:
        dict_coverage_database (dict): A dictionary containing the data of the neutron star population falling
            in the survey sky coverage.

    Returns:
        (dict): A dictionary containing the initialized magneto-rotational properties of the neutron star population
            in the survey sky coverage.
    """

    age = dict_coverage_database["age"]

    # Initialize neutron star population properties.
    pop_initial = ipop.InitialNeutronStarPopulation(NS_number=len(age))

    # Computing the initial field strengths, misalignment angles, and periods.
    B_initial = pop_initial.magnetic_field()
    chi_initial = pop_initial.misalignment_angle()
    P_initial = pop_initial.period()

    dictionary_initial_pop_magrot = {
        "age": age,
        "B_initial": B_initial,
        "chi_initial": chi_initial,
        "P_initial": P_initial,
    }

    return dictionary_initial_pop_magrot


def evolve_population_magrot(
    dict_pop_initial_magrot: dict,
    output_path: pathlib.Path,
) -> dict:
    """
    Evolve the magneto-rotational properties of a neutron star population over time based on initial conditions.

    Args:
        dict_pop_initial_magrot (dict): Dictionary containing initial magneto-rotational properties of the population.
        output_path (pathlib.Path): The path where the evolution data will be saved if enabled in the configuration.

    Returns:
        (dict): A dictionary containing the properties of the evolved neutron star population.
    """
    age = dict_pop_initial_magrot["age"]
    B_initial = dict_pop_initial_magrot["B_initial"]
    chi_initial = dict_pop_initial_magrot["chi_initial"]
    P_initial = dict_pop_initial_magrot["P_initial"]

    a_late = cfg["a_late"]

    # Determine the evolved magnetic field, misalignment angle and rotation period.
    (
        B_final,
        chi_final,
        P_final,
        magrot_evol_dict,
    ) = mre.magneto_rotational_evolution(
        B_initial, chi_initial, P_initial, age, a_late
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

    # Determining the final period derivative.
    period_derivative_vect = np.vectorize(pdv.period_derivative)
    P_dot_final = (
        period_derivative_vect(
            B_final,
            chi_final,
            P_final,
        )
        / const.YR_TO_S
    )

    dictionary_final_pop_magrot = {
        "B_initial": B_initial,
        "B": B_final,
        "chi": chi_final,
        "P": P_final,
        "P_dot": P_dot_final,
    }

    return dictionary_final_pop_magrot


def radio_intercepted(dict_final_pop: dict) -> dict:
    """
    Filter and compute properties of neutron stars whose radio beams intercept our line of sight.

    Args:
        dict_final_pop (dict): Dictionary containing the evolved properties of a neutron star population.

    Returns:
        (dict): A dictionary containing properties of the neutron stars whose radio beams intercept our line of sight.
    """

    # Select only the stars that can be detected in radio by the considered surveys.
    coverage_radio = dict_final_pop["coverage_radio"]
    dict_final_pop_filtered = {
        key: value[coverage_radio] for key, value in dict_final_pop.items()
    }

    # Collect the coverage data for each radio survey.
    coverage_data = {}
    for key in dict_final_pop.keys():
        if key.startswith("coverage_radio_"):
            coverage_data[key] = dict_final_pop[key][coverage_radio]

    # Find the pulsars whose radio beam intercepts our line of sight and compute the intrinsic properties
    # of their radio emission.
    (
        intercepted_radio,
        w_int_s,
        L_radio_bol,
        S_radio_bol,
        spectral_index,
        DM,
        tau_sc,
    ) = er.calculate_radio_emission(
        dict_final_pop_filtered["P"],
        dict_final_pop_filtered["P_dot"],
        dict_final_pop_filtered["age"],
        dict_final_pop_filtered["l"],
        dict_final_pop_filtered["b"],
        dict_final_pop_filtered["dist"],
        dict_final_pop_filtered["chi"],
    )

    dictionary_intercepted_radio = {
        key: value[intercepted_radio]
        for key, value in dict_final_pop_filtered.items()
    }
    dictionary_intercepted_radio["w_int"] = w_int_s
    dictionary_intercepted_radio["L_radio_bol"] = L_radio_bol
    dictionary_intercepted_radio["S_radio_bol"] = S_radio_bol
    dictionary_intercepted_radio["spectral_index"] = spectral_index
    dictionary_intercepted_radio["DM"] = DM
    dictionary_intercepted_radio["tau_sc"] = tau_sc

    return dictionary_intercepted_radio


def update_detected_dictionary(
    dict_to_update: dict, detected_mask: np.ndarray, **kwargs: np.ndarray
) -> dict:
    """
    This function updates a dictionary to include only the elements corresponding to the given detected mask.

    Args:
        dict_to_update (dict): The original dictionary containing properties of neutron stars.
        detected_mask (np.ndarray): boolean mask indicating which neutron stars in `dict_to_update` are detected.
        **kwargs (np.ndarray): Additional property values provided as keyword arguments.
            These properties will also be filtered using the `detected_mask`.

    Returns:
        (dict): A new dictionary containing only the detected neutron stars from `dict_to_update` and combining
            the keys already present in the original dictionary with the ones provided in `kwargs`.
    """

    # Extract keys from the dictionary that has to be updated.
    properties = list(dict_to_update.keys())

    # Filtering the detected stars in the original dictionary.
    filtered_dict = {
        prop: dict_to_update[prop][detected_mask].tolist()
        for prop in properties
    }
    # Add and filter the properties specified in `kwargs` to another dictionary.
    additional_filtered_dict = {
        key: value[detected_mask].tolist() for key, value in kwargs.items()
    }

    # Combine the two dictionaries.
    combined_dict = filtered_dict | additional_filtered_dict

    return combined_dict


def radio_detection(
    radio_surveys: dict, dictionary_intercepted_radio: dict
) -> dict:
    """
    Simulate radio detections for various surveys and update the dictionaries with the properties
    of detected neutron stars.

    Args:
        radio_surveys (dict): Dictionary containing the radio survey objects.
        dictionary_intercepted_radio (dict): Dictionary with properties of intercepted radio pulsars.

    Returns:
        (dict): A dictionary containing the properties of detected pulsars for each survey.
    """
    # Initialize variables for HTRU_low and HTRU_mid to avoid "referenced before assignment".
    detected_HTRU_low = np.array([])
    w_eff_low = None
    S_radio_obs_mean_low = None
    S_radio_obs_mean_1400_low = None

    detected_HTRU_mid = np.array([])
    w_eff_mid = None
    S_radio_obs_mean_mid = None
    S_radio_obs_mean_1400_mid = None

    # Process each survey.
    detected_dictionaries = {}
    for survey_name in radio_surveys:
        (
            detected_mask,
            w_eff,
            S_radio_obs_mean,
            S_radio_obs_mean_1400,
        ) = radio_surveys[survey_name].detected_radio_population(
            dictionary_intercepted_radio["w_int"],
            dictionary_intercepted_radio["DM"],
            dictionary_intercepted_radio["P"],
            dictionary_intercepted_radio["age"],
            dictionary_intercepted_radio[f"coverage_radio_{survey_name}"],
            dictionary_intercepted_radio["l"],
            dictionary_intercepted_radio["b"],
            dictionary_intercepted_radio["S_radio_bol"],
            dictionary_intercepted_radio["spectral_index"],
            dictionary_intercepted_radio["tau_sc"],
        )
        detected_dictionaries[survey_name] = update_detected_dictionary(
            dictionary_intercepted_radio,
            detected_mask,
            w_eff=w_eff,
            S_radio_obs_mean=S_radio_obs_mean,
            S_radio_obs_mean_1400=S_radio_obs_mean_1400,
        )

        # Save the properties for the HTRU low and mid surveys separately.
        if survey_name == "HTRU_low":
            detected_HTRU_low = detected_mask
            w_eff_low = w_eff
            S_radio_obs_mean_low = S_radio_obs_mean
            S_radio_obs_mean_1400_low = S_radio_obs_mean_1400

        elif survey_name == "HTRU_mid":
            detected_HTRU_mid = detected_mask
            w_eff_mid = w_eff
            S_radio_obs_mean_mid = S_radio_obs_mean
            S_radio_obs_mean_1400_mid = S_radio_obs_mean_1400

    if (
        "HTRU_low" in radio_surveys.keys()
        and "HTRU_mid" in radio_surveys.keys()
    ):
        # Since the sky coverage of the HTRU mid and low surveys overlap, we remove those stars from the mid
        # survey that are already in the low survey in order to not double count individual objects.
        detected_HTRU_low_mid = detected_HTRU_low | detected_HTRU_mid
        detected_dictionaries["HTRU_low_mid"] = update_detected_dictionary(
            dictionary_intercepted_radio,
            detected_HTRU_low_mid,
            w_eff=np.where(detected_HTRU_low, w_eff_low, w_eff_mid),
            S_radio_obs_mean=np.where(
                detected_HTRU_low, S_radio_obs_mean_low, S_radio_obs_mean_mid
            ),
            S_radio_obs_mean_1400=np.where(
                detected_HTRU_low,
                S_radio_obs_mean_1400_low,
                S_radio_obs_mean_1400_mid,
            ),
            HTRU_low=detected_HTRU_low,
            HTRU_mid=detected_HTRU_mid,
            idx=dictionary_intercepted_radio["idx"],
        )

        # Remove the single HTRU low and mid surveys.
        del detected_dictionaries["HTRU_low"]
        del detected_dictionaries["HTRU_mid"]

    return detected_dictionaries


def x_detection(
    dict_final_pop: dict,
    L_x_interpolator: RectBivariateSpline,
    L_x_threshold: float,
    age_cutoff: float,
    S_x_abs_threshold: float,
) -> dict:
    """
    Detects neutron stars based on X-ray luminosity and updates their properties.

    Args:
        dict_final_pop (dict): Dictionary containing the properties of the evolved neutron star population.
        L_x_interpolator (RectBivariateSpline): Interpolator used to calculate thermal X-ray luminosity based on age and magnetic field.
        L_x_threshold (float): A lower limit for the X-ray luminosity.
        age_cutoff (float): An upper limit for the neutron star age for X-ray detection.
        S_x_abs_threshold (float): The absorbed flux threshold for X-ray detection.

    Returns:
        (dict): A dictionary containing the properties of detected neutron stars in X-rays.
    """
    # Select only the stars that can be detected in X-rays by the considered surveys.
    coverage_x = dict_final_pop["coverage_x"]
    dict_final_pop_filtered = {
        key: value[coverage_x] for key, value in dict_final_pop.items()
    }

    # Compute the properties of the X-ray bright neutron stars.
    xray_bright_mask, L_x_therm, S_x_abs, N_H = ex.calculate_x_emission(
        dict_final_pop_filtered["B"],
        dict_final_pop_filtered["B_initial"],
        dict_final_pop_filtered["age"],
        dict_final_pop_filtered["ra"],
        dict_final_pop_filtered["dec"],
        dict_final_pop_filtered["dist"],
        L_x_interpolator,
        L_x_threshold,
        age_cutoff,
    )

    dict_xray_bright = {
        key: value[xray_bright_mask]
        for key, value in dict_final_pop_filtered.items()
    }
    dict_xray_bright["L_x_therm"] = L_x_therm
    dict_xray_bright["S_x_abs"] = S_x_abs
    dict_xray_bright["N_H"] = N_H

    # Apply a flux threshold to mimic detection biases.
    detected_mask = sx.detected_x_population_flux_threshold(
        dict_xray_bright["S_x_abs"],
        S_x_abs_threshold,
    )

    dictionary_detected = update_detected_dictionary(
        dict_xray_bright, detected_mask
    )

    return dictionary_detected


def build_dataframe(
    data_dict: dict, parameters: list, units: list
) -> pd.DataFrame:
    """
    Helper function to create a DataFrame with a MultiIndex header.

    Args:
        data_dict (dict): A dictionary containing the data to be saved in the dataframe.
        parameters (list): A list of parameter names, used as the first level of the MultiIndex header.
        units (list): A list of physical units, used as the second level of the MultiIndex header.

    Returns:
        (pd.DataFrame): A Pandas DataFrame with a MultiIndex header, where columns are
            indexed by parameters and units.
    """
    # If the key `"idx"` is present, it is removed.
    data_dict.pop("idx", None)

    header = pd.MultiIndex.from_arrays([parameters, units])

    df = pd.DataFrame.from_dict(data=data_dict)
    df.columns = header

    return df


def create_output_dataframe(
    dictionary_detected_radio: dict,
    dictionary_detected_x: dict,
    survey_x: bool,
) -> dict:
    """
    Creates Pandas DataFrames containing the information on the detected neutron stars for each survey.

    Args:
        dictionary_detected_radio (dict): Dictionary containing detected neutron star properties for each radio survey.
        dictionary_detected_x (dict): Dictionary containing detected neutron star properties for an X-ray survey.
        survey_x (bool): Boolean flag to indicate whether or not saving the results for a X-ray survey.

    Returns:
        dict: A dictionary of DataFrames, one for each survey containing detected neutron stars' information.
    """

    # Defining the parameters and units that are common for all surveys.
    parameters_radio = [
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
        "S_radio_obs_mean_1400",
        "w_int",
        "w_eff",
        "spectral_index",
    ]
    units_radio = [
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
        "[Jy]",
        "[s]",
        "[s]",
        "",
    ]

    # Initialize an empty dictionary to store the resulting DataFrames.
    dfs = {}

    # Loop over each survey's detected dictionary and generate the corresponding DataFrame.
    for survey_name, survey_data in dictionary_detected_radio.items():
        # Check if the survey is a combined survey like 'HTRU_low_mid'.
        if survey_name == "HTRU_low_mid":
            parameters_survey = parameters_radio + ["HTRU_low", "HTRU_mid"]
            units_survey = units_radio + ["", ""]
        else:
            parameters_survey = parameters_radio
            units_survey = units_radio

        # Build the DataFrame using the appropriate parameters and units.
        df = build_dataframe(survey_data, parameters_survey, units_survey)
        dfs[
            survey_name
        ] = df  # Store the DataFrame in the dictionary with survey_name as key.

    if survey_x:
        parameters_x = [
            "age",
            "RA",
            "DEC",
            "l",
            "b",
            "N_H",
            "d",
            "pm_RA",
            "pm_DEC",
            "v_ls",
            "B",
            "chi",
            "P",
            "P_dot",
            "L_x_therm",
            "S_x_abs",
        ]
        units_x = [
            "[yr]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[cm^-2]",
            "[kpc]",
            "[mas yr^-1]",
            "[mas yr^-1]",
            "[km s^-1]",
            "[G]",
            "[rad]",
            "[s]",
            "[s s^-1]",
            "[erg s^-1]",
            "[erg s^-1 cm^-2]",
        ]

        # Build the DataFrame using the appropriate parameters and units.
        df = build_dataframe(dictionary_detected_x, parameters_x, units_x)
        dfs[
            "X-ray"
        ] = df  # Store the DataFrame in the dictionary with survey_name as key.

    return dfs


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
        # Initialize radio surveys.
        surveys_radio, dictionary_detected_radio = initialize_radio_surveys()
        surveys_radio_cfg = cfg["surveys_radio"]

        (
            dictionary_detected_x,
            luminosity_x_interpolator,
        ) = initialize_x_surveys()

        # Initialize the indicator for an excess in birth rate to False.
        cfg["birth_rate_excess"] = False

        # This variable progressively count how many stars we create in total.
        n_created = 0
        # In these dictionaries we save how many stars we progressively detect in total in each survey and
        # the percentage related to the real detected numbers.
        n_detected_sim = {survey: 0 for survey in surveys_radio_cfg}
        n_detected_sim_x = 0
        percentage_detected = {survey: 0 for survey in surveys_radio_cfg}

        # In this dictionary we save how many stars we have created to reach the desirable number in each survey.
        n_created_at_match = {survey: 0 for survey in surveys_radio_cfg}
        # In this dictionary we save how many stars we have detected when we reach the desirable number in each survey.
        n_detected_sim_at_match = {survey: 0 for survey in surveys_radio_cfg}

        # List to store indices of detected neutron stars to avoid resampling.
        idx_remove = []

        # To speed up the simulation, generate new neutron stars in batches.
        n_batchsize = 100000

        # Flags for adjusting batch size as we approach the target detection number for all surveys.
        flags = {0.9: False, 0.95: False}

        # Flags for stopping the simulation as we reach the target detection number for all surveys.
        stop_flags = {survey: False for survey in surveys_radio_cfg}

        # Loop to simulate stars until the detected number of pulsars for all surveys is reached.
        while not all(stop_flags.values()):

            with timewith.TimeWith(
                "[LoadPopulationDynamics]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):
                # ===================== POPULATION INITIALIZATION ========================
                # Evaluate the percentage of neutron stars detected by the simulated
                # surveys with respect to the real surveys and adjust the batch size accordingly.
                for survey in surveys_radio_cfg:
                    percentage_detected[survey] = (
                        n_detected_sim[survey]
                        / surveys_radio_cfg[survey]["detected_real"]
                    )

                if (
                    all(value > 0.9 for value in percentage_detected.values())
                    and not flags[0.9]
                ):
                    flags[0.9] = True
                    n_batchsize = 10000
                elif (
                    all(value > 0.95 for value in percentage_detected.values())
                    and not flags[0.95]
                ):
                    flags[0.95] = True
                    n_batchsize = 5000

                # Update the total number of simulated neutron stars.
                n_created += n_batchsize
                log.info(f"Total number of created neutron stars: {n_created}")

                # Load a batch of dynamically evolved neutron stars from the dynamical database.
                database_dyn_batch = load_database_dyn(
                    args.dyn_data,
                    n_batchsize,
                    idx_remove,
                )

                # Compute the maximum simulation time in centuries.
                t_max = cfg["t_age_max"] / 100

                # Filter the loaded database batch with the surveys' sky coverage.
                database_coverage, idx_remove = apply_surveys_coverage(
                    surveys_radio,
                    cfg["survey_xray"],
                    database_dyn_batch,
                    idx_remove,
                    dist_cutoff=35.0,
                )

                # Initialize neutron star magneto-rotational properties.
                pop_magrot_initial = initialize_population_magrot(
                    database_coverage
                )

            with timewith.TimeWith(
                "[SimulateMagnetoRotationalEvolution]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================
                # Evolve in time the magneto-rotational properties.
                pop_magrot_final = evolve_population_magrot(
                    pop_magrot_initial, output_path
                )

                # Merge the dictionary containing the final magneto-rotational properties with the filtered dynamical
                # database.
                pop_final = database_coverage | pop_magrot_final

            with timewith.TimeWith(
                "[SimulateRadioDetection]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                # ===================== RADIO DETECTION ========================
                # Filter the population to include only pulsars whose radio beam intercepts our line of sight.
                pop_intercepted_radio = radio_intercepted(pop_final)
                if len(pop_intercepted_radio["age"]) == 0:
                    break

                # Filter the population to include only detected pulsars by the radio surveys.
                pop_detected_radio_update = radio_detection(
                    surveys_radio, pop_intercepted_radio
                )

                for survey in surveys_radio_cfg:
                    # Check the number of detected pulsars for each survey.
                    n_detected_sim[survey] += len(
                        pop_detected_radio_update[survey]["age"]
                    )
                    log.info(
                        f"Total number of neutron stars detected by {survey}: {n_detected_sim[survey]}"
                    )

                    # If the number of simulated detected pulsars matches the real one, store the value of created neutron stars.
                    if (
                        n_detected_sim[survey]
                        >= surveys_radio_cfg[survey]["detected_real"]
                        and not stop_flags[survey]
                    ):
                        n_detected_sim_at_match[survey] = n_detected_sim[
                            survey
                        ]
                        stop_flags[survey] = True
                        n_created_at_match[survey] = n_created

                    # Update the detection dictionary.
                    dictionary_detected_radio[survey] = {
                        key: value + pop_detected_radio_update[survey][key]
                        for key, value in dictionary_detected_radio[
                            survey
                        ].items()
                    }

                    # Remove from the dynamical database the stars that have been detected.
                    idx_det_radio = list(
                        dictionary_detected_radio[survey]["idx"]
                    )
                    idx_remove += idx_det_radio

            # ===================== X DETECTION ========================

            if cfg["survey_xray"]:
                with timewith.TimeWith(
                    "[SimulateXrayDetection]",
                    cfg["profile_log"],
                    cfg["profile_json"],
                    cfg["show_profiling"],
                ):
                    # Filter the population to include only detected pulsars by the X-ray surveys.
                    pop_detected_x_update = x_detection(
                        pop_final,
                        luminosity_x_interpolator,
                        L_x_threshold=1.0e30,
                        S_x_abs_threshold=1.0e-15,
                        age_cutoff=1.0e6,
                    )

                    # Check the number of detected pulsars for each survey.
                    n_detected_sim_x += len(pop_detected_x_update["age"])
                    log.info(
                        f"Total number of neutron stars detected in X-rays: {n_detected_sim_x}"
                    )

                    # Update the detection dictionary.
                    dictionary_detected_x = {
                        key: value + pop_detected_x_update[key]
                        for key, value in dictionary_detected_x.items()
                    }

                    # Remove from the dynamical database the stars that have been detected.
                    idx_det_x = list(dictionary_detected_x["idx"])
                    idx_remove += idx_det_x

            # ==========================================================

            # Compute the total current birth rate in NSs per century.
            birth_rate = n_created / t_max
            log.info(
                f"Galactic neutron star birth rate per century: {birth_rate} neutron stars per century."
            )
            # If the current birth rate exceeds an upper limit of 5 NS per century stop the simulation.
            if birth_rate > 5:
                log.info(
                    "Simulation stopped! Galactic neutron star birth rate exceeds 5 neutron stars per century."
                )
                break

        # Compute the neutron star birth rate for each survey.
        birth_rates = {}
        for survey in dictionary_detected_radio:
            birth_rates[survey] = n_created_at_match[survey] / t_max

            log.info(
                f"Galactic neutron star birth rate per century according to {survey}: {birth_rates[survey]} neutron stars per century."
            )

            # Add the information of the birth rates and the number of detected neutron star to the configuration file.
            cfg[f"birth_rate_{survey}_at_match"] = birth_rates[survey]
            cfg[f"n_detected_sim_{survey}_at_match"] = n_detected_sim_at_match[
                survey
            ]
            cfg[f"n_detected_sim_{survey}_tot"] = n_detected_sim[survey]

        # ===================== EXPORT OUTPUT ========================
        with timewith.TimeWith(
            "[Export]",
            cfg["profile_log"],
            cfg["profile_json"],
            cfg["show_profiling"],
        ):
            log.info("Creating data frame for exporting...")

            # Create output dataframes for each survey.
            dfs = create_output_dataframe(
                dictionary_detected_radio,
                dictionary_detected_x,
                cfg["survey_xray"],
            )

            # Save the data frame as a compressed binary file.
            for survey in dictionary_detected_radio:
                output_path_survey = (
                    output_path / f"survey_{survey}_results.pkl.gz"
                )
                dfs[survey].to_pickle(output_path_survey, compression="gzip")
                log.info(
                    f"Output of the detected population with {survey} generated in {os.getcwd()}/{output_path_survey}"
                )

            if cfg["survey_xray"]:
                output_path_survey = output_path / "survey_xray_results.pkl.gz"
                dfs["X-ray"].to_pickle(output_path_survey, compression="gzip")
                log.info(
                    f"Output of the detected population with a X-ray survey generated in {os.getcwd()}/{output_path_survey}"
                )

            # Dump updated configuration to output path.
            config_dump_path = pathlib.Path(output_path) / "configuration.json"
            with open(config_dump_path, "w") as f:
                json.dump(cfg, f, indent=4, sort_keys=True)


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
        default="output/sim_magrot",
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
