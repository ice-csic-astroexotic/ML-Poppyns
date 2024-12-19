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
import sys
import time
from typing import List, Tuple

import numpy as np
import orjson
import pandas as pd

import pypopsyn.simulator.config_simulator as configuration
import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.benchmark.timewith as timewith
import utilities.samplers.memory_efficient_sampling as mes
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

# Suppressing healpy related logging output.
logging.getLogger("healpy").setLevel(logging.WARNING)


def initialize_radio_surveys(cfg: dict) -> Tuple[dict, dict, dict, dict, dict]:
    """
    Initialize and return radio survey objects and empty detection dictionaries.

    Args:
        cfg (dict): A dictionary containing the simulator configuration settings.

    Returns:
        (Tuple[dict, dict, dict, dict, dict]): A tuple of dictionaries containing the following information:
            - A dictionary of initialized radio survey objects, keyed by survey names (e.g., "PMPS", "SMPS", etc.).
            - An empty dictionary for storing detected neutron star data for the "PMPS" survey.
            - An empty dictionary for storing detected neutron star data for the "SMPS" survey.
            - An empty dictionary for storing detected neutron star data for the "HTRU_low" and "HTRU_mid" surveys.
            - An empty dictionary for storing detected neutron star data for the "HTRU_high" survey.
    """
    PMPS_par_path = pathlib.Path(cfg["path_to_software"]).joinpath(
        "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json"
    )
    SMPS_par_path = pathlib.Path(cfg["path_to_software"]).joinpath(
        "pypopsyn/simulator/multiband_surveys/Swinburne_Parkes_parameters.json"
    )
    HTRU_low_par_path = pathlib.Path(cfg["path_to_software"]).joinpath(
        "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json"
    )
    HTRU_mid_par_path = pathlib.Path(cfg["path_to_software"]).joinpath(
        "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json"
    )
    HTRU_high_par_path = pathlib.Path(cfg["path_to_software"]).joinpath(
        "pypopsyn/simulator/multiband_surveys/htru_high_parameters.json"
    )

    # Initialize and return surveys as a dictionary.
    radio_surveys = {
        "PMPS": sr.SurveyRadio(PMPS_par_path),
        "SMPS": sr.SurveyRadio(SMPS_par_path),
        "HTRU_low": sr.SurveyRadio(HTRU_low_par_path),
        "HTRU_mid": sr.SurveyRadio(HTRU_mid_par_path),
        "HTRU_high": sr.SurveyRadio(HTRU_high_par_path),
    }

    # Initializing the dictionaries where we save the detected neutron stars for each survey.
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

    # Specialized template for HTRU detection dictionaries
    htru_low_mid_detection_template = detection_template.copy()
    htru_low_mid_detection_template.update({"HTRU_low": [], "HTRU_mid": []})

    # Create detection dictionaries
    dictionary_detected_PMPS = detection_template.copy()
    dictionary_detected_SMPS = detection_template.copy()
    dictionary_detected_HTRU_low_mid = htru_low_mid_detection_template.copy()
    dictionary_detected_HTRU_high = detection_template.copy()

    return (
        radio_surveys,
        dictionary_detected_PMPS,
        dictionary_detected_SMPS,
        dictionary_detected_HTRU_low_mid,
        dictionary_detected_HTRU_high,
    )


def load_database_dyn(
    dyn_path_pop: pathlib.Path,
    n_batchsize: int,
    NS_number: int,
    idx_remove: list,
) -> dict:
    """
    Load a chunk of a dynamically evolved population from a .csv file, convert coordinates and return a dictionary
    with the dynamical information of the selected stars.

    Args:
        dyn_path_pop (pathlib.Path): Path to the file containing the dynamically evolved population data.
        n_batchsize (int): Batch size of stars to select when loading the data.
        NS_number (int): The total number of neutron stars in the dynamical database.
        idx_remove (list): List of indices to remove from the dynamical database.

    Returns:
        (dict): A dictionary containing the data of the selected dynamical population chunk.
    """

    # Load the chunk of the file containing the dynamically evolved population parameters.
    df_dyn = mes.select(
        dyn_path_pop,
        n_batchsize,
        NS_number,
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
        "dist_heliocentric": dist_heliocentric_icrs,
        "pm_ra": pm_ra,
        "pm_dec": pm_dec,
        "v_ls": v_ls_icrs,
        "idx": df_dyn.index.values,
    }

    return dictionary_dyn_database_chunk


def apply_surveys_coverage(
    radio_surveys: dict,
    dyn_database_dict: dict,
    idx_remove: list,
    dist_cutoff: float,
) -> Tuple[dict, list]:
    """
    Apply survey coverage criteria to filter a dynamic population dataset based on sky coverage,
    distance, and age cutoffs, and update the indices of entries to be removed.

    Args:
        radio_surveys (dict): A dictionary of radio survey objects, containing the information on the sky coverage.
        dyn_database_dict (dict): A dictionary containing the data of a dynamical population.
        idx_remove (list): A list of indices of entries to be removed based on the filtering criteria.
        dist_cutoff (float): The maximum heliocentric distance to include in the survey coverage.

    Returns:
        (Tuple[dict, list]):
            - A dictionary containing data for stars that meet the coverage criteria.
            - An updated list of indices of stars that are outside the coverage and should be removed.
    """

    age = dyn_database_dict["age"]
    ra = dyn_database_dict["ra"]
    dec = dyn_database_dict["dec"]
    l_gal = dyn_database_dict["l"]
    b_gal = dyn_database_dict["b"]
    dist = dyn_database_dict["dist_heliocentric"]
    pm_ra = dyn_database_dict["pm_ra"]
    pm_dec = dyn_database_dict["pm_dec"]
    v_ls = dyn_database_dict["v_ls"]
    idx = dyn_database_dict["idx"]

    dist_mask = dist < dist_cutoff

    # Select only neutron stars that fall into the sky region covered by the surveys.
    survey_names = list(radio_surveys.keys())
    coverage = {}

    for name in survey_names:
        coverage[name] = radio_surveys[name].sky_coverage(
            ra, dec, l_gal, b_gal
        )

    # Determine which stars fall into the sky region covered by any of the considered radio surveys.
    coverage_radio = (
        functools.reduce(
            lambda a, b: a | b, (coverage[name] for name in survey_names)
        )
    ) & dist_mask

    coverage_tot = coverage_radio

    dictionary_coverage_database = {
        "age": age[coverage_tot],
        "ra": ra[coverage_tot],
        "dec": dec[coverage_tot],
        "l": l_gal[coverage_tot],
        "b": b_gal[coverage_tot],
        "dist": dist[coverage_tot],
        "pm_ra": pm_ra[coverage_tot],
        "pm_dec": pm_dec[coverage_tot],
        "v_ls": v_ls[coverage_tot],
        "idx": idx[coverage_tot],
        "coverage_radio": coverage_radio[coverage_tot],
        "coverage_PMPS": coverage["PMPS"][coverage_tot],
        "coverage_SMPS": coverage["SMPS"][coverage_tot],
        "coverage_HTRU_low": coverage["HTRU_low"][coverage_tot],
        "coverage_HTRU_mid": coverage["HTRU_mid"][coverage_tot],
        "coverage_HTRU_high": coverage["HTRU_high"][coverage_tot],
    }

    # Remove stars that do not fall into the total sky coverage.
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

    # Initialize neutron star population properties.
    pop_initial = ipop.InitialNeutronStarPopulation(
        NS_number=len(dict_coverage_database["age"])
    )

    # Computing the initial field strengths, misalignment angles, and periods.
    B_initial = pop_initial.magnetic_field()
    chi_initial = pop_initial.misalignment_angle()
    P_initial = pop_initial.period()

    dictionary_initial_pop_magrot = {
        "B_initial": B_initial,
        "chi_initial": chi_initial,
        "P_initial": P_initial,
    }

    return dictionary_initial_pop_magrot


def evolve_population_magrot(
    cfg: dict,
    dict_pop_initial_magrot: dict,
    dict_coverage_database: dict,
    output_path: pathlib.Path,
) -> dict:
    """
    Evolve the magneto-rotational properties of a neutron star population over time based on initial conditions.

    Args:
        cfg (dict): A dictionary containing the simulator configuration settings.
        dict_pop_initial_magrot (dict): Dictionary containing initial magneto-rotational properties of the population.
        dict_coverage_database (dict): Dictionary containing the dynamical properties of the neutron star population
            falling in the survey sky coverage.
        output_path (pathlib.Path): The path where the evolution data will be saved if enabled in the configuration.

    Returns:
        (dict): A dictionary containing the properties of the evolved neutron star population.
    """
    age = dict_coverage_database["age"]
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

    dictionary_final_pop_magrot = {
        "B_initial": B_initial,
        "B_final": B_final,
        "chi_final": chi_final,
        "P_final": P_final,
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

    coverage_radio = dict_final_pop["coverage_radio"]

    # Select only the stars that can be detected in radio by the considered surveys.
    age = dict_final_pop["age"][coverage_radio]
    l_gal = dict_final_pop["l"][coverage_radio]
    b_gal = dict_final_pop["b"][coverage_radio]
    ra = dict_final_pop["ra"][coverage_radio]
    dec = dict_final_pop["dec"][coverage_radio]
    dist = dict_final_pop["dist"][coverage_radio]
    pm_ra = dict_final_pop["pm_ra"][coverage_radio]
    pm_dec = dict_final_pop["pm_dec"][coverage_radio]
    v_ls = dict_final_pop["v_ls"][coverage_radio]
    P = dict_final_pop["P_final"][coverage_radio]
    B = dict_final_pop["B_final"][coverage_radio]
    chi = dict_final_pop["chi_final"][coverage_radio]
    idx = dict_final_pop["idx"][coverage_radio]
    coverage_PMPS = dict_final_pop["coverage_PMPS"][coverage_radio]
    coverage_SMPS = dict_final_pop["coverage_SMPS"][coverage_radio]
    coverage_HTRU_low = dict_final_pop["coverage_HTRU_low"][coverage_radio]
    coverage_HTRU_mid = dict_final_pop["coverage_HTRU_mid"][coverage_radio]
    coverage_HTRU_high = dict_final_pop["coverage_HTRU_high"][coverage_radio]

    # Find the pulsars whose radio beam intercepts our line of sight and compute the intrinsic properties
    # of their radio emission.
    dictionary_intercepted_radio = er.calculate_radio_emission(
        P,
        age,
        l_gal,
        b_gal,
        dist,
        B,
        chi,
        idx,
    )

    intercepted_radio = dictionary_intercepted_radio["intercepted_radio"]

    dictionary_intercepted_radio = {
        "age": dictionary_intercepted_radio["age"],
        "l": dictionary_intercepted_radio["l"],
        "b": dictionary_intercepted_radio["b"],
        "ra": ra[intercepted_radio],
        "dec": dec[intercepted_radio],
        "dist": dist[intercepted_radio],
        "pm_ra": pm_ra[intercepted_radio],
        "pm_dec": pm_dec[intercepted_radio],
        "v_ls": v_ls[intercepted_radio],
        "B": dictionary_intercepted_radio["B"],
        "chi": dictionary_intercepted_radio["chi"],
        "P": dictionary_intercepted_radio["P"],
        "P_dot": dictionary_intercepted_radio["P_dot"],
        "w_int": dictionary_intercepted_radio["w_int_s"],
        "DM": dictionary_intercepted_radio["DM"],
        "idx": dictionary_intercepted_radio["idx"],
        "L_radio_bol": dictionary_intercepted_radio["L_radio_bol"],
        "S_radio_bol": dictionary_intercepted_radio["S_radio_bol"],
        "spectral_index": dictionary_intercepted_radio["spectral_index"],
        "tau_sc": dictionary_intercepted_radio["tau_sc"],
        "coverage_PMPS": coverage_PMPS[intercepted_radio],
        "coverage_SMPS": coverage_SMPS[intercepted_radio],
        "coverage_HTRU_low": coverage_HTRU_low[intercepted_radio],
        "coverage_HTRU_mid": coverage_HTRU_mid[intercepted_radio],
        "coverage_HTRU_high": coverage_HTRU_high[intercepted_radio],
    }

    return dictionary_intercepted_radio


# Function to update detected neutron star dictionary
def update_detected_dictionary(
    dict_to_update: dict, detected_mask: list, **kwargs: np.ndarray
) -> dict:
    """
    Updates a dictionary to include only the elements corresponding to detected indices.

    Args:
        dict_to_update (dict): The original dictionary containing properties of neutron stars.
            Each key corresponds to a property (e.g., 'w_eff', 'S_radio_obs_mean') and its values are lists of those properties.
        detected_mask (list): boolean mask indicating which neutron stars in `dict_to_update` are detected.
        **kwargs (np.ndarray): Additional property values provided as keyword arguments.
            These properties will also be filtered using the `detected_indices`.

    Returns:
        (dict): A new dictionary containing only the detected neutron stars from `dict_to_update` and combining
            the keys already present in the original dictionary with the one provided in `kwargs`.
    """

    # Extract keys from the dictionary that has to be updated.
    properties = list(dict_to_update.keys())

    print(np.shape(dict_to_update["age"]))
    # Filtering the detected stars in the original dictionary and adding the properties specified in `kwargs`.
    combined_dict = {
        prop: dict_to_update[prop][detected_mask].tolist()
        for prop in properties
    } | {key: value[detected_mask].tolist() for key, value in kwargs.items()}

    return combined_dict


def radio_detection(
    radio_surveys: dict, dictionary_intercepted_radio: dict
) -> Tuple[dict, dict, dict, dict]:
    """
    Simulate radio detections for various surveys and update the dictionaries with the properties
    of detected neutron stars.

    Args:
        radio_surveys (dict): Dictionary containing the radio survey objects.
        dictionary_intercepted_radio (dict): Dictionary with properties of intercepted radio pulsars.

    Returns:
        (Tuple[dict, dict, dict, dict]): A tuple consisting of the dictionaries containing the properties
            of detected pulsars for each survey.
    """

    # Process each survey dynamically
    detected_dictionaries = {}
    for survey_key in radio_surveys.keys():
        (
            detected_mask,
            w_eff,
            S_radio_obs_mean,
            S_radio_obs_mean_1400,
        ) = radio_surveys[survey_key].detected_radio_population(
            dictionary_intercepted_radio["w_int"],
            dictionary_intercepted_radio["DM"],
            dictionary_intercepted_radio["P"],
            dictionary_intercepted_radio["age"],
            dictionary_intercepted_radio[f"coverage_{survey_key}"],
            dictionary_intercepted_radio["l"],
            dictionary_intercepted_radio["b"],
            dictionary_intercepted_radio["S_radio_bol"],
            dictionary_intercepted_radio["spectral_index"],
            dictionary_intercepted_radio["tau_sc"],
        )
        detected_dictionaries[survey_key] = update_detected_dictionary(
            dictionary_intercepted_radio,
            detected_mask,
            w_eff=w_eff,
            S_radio_obs_mean=S_radio_obs_mean,
            S_radio_obs_mean_1400=S_radio_obs_mean_1400,
        )

        if survey_key == "HTRU_low":
            detected_HTRU_low = detected_mask
            w_eff_low = w_eff
            S_radio_obs_mean_low = S_radio_obs_mean
            S_radio_obs_mean_1400_low = S_radio_obs_mean_1400

        elif survey_key == "HTRU_mid":
            detected_HTRU_mid = detected_mask
            w_eff_mid = w_eff
            S_radio_obs_mean_mid = S_radio_obs_mean
            S_radio_obs_mean_1400_mid = S_radio_obs_mean_1400

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

    return (
        detected_dictionaries["PMPS"],
        detected_dictionaries["SMPS"],
        detected_dictionaries["HTRU_low_mid"],
        detected_dictionaries["HTRU_high"],
    )


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
    data_dict.pop("idx", None)  # Remove idx_det if present

    header = pd.MultiIndex.from_arrays([parameters, units])

    df = pd.DataFrame.from_dict(data=data_dict)
    df.columns = header

    return df


def create_output_dataframe(
    dictionary_detected_PMPS: dict,
    dictionary_detected_SMPS: dict,
    dictionary_detected_HTRU_low_mid: dict,
    dictionary_detected_HTRU_high: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates Pandas DataFrames containing the information on the detected neutron stars in each survey.

    Args:
        dictionary_detected_PMPS (dict): Dictionary containing detected PMPS neutron star properties.
        dictionary_detected_SMPS (dict): Dictionary containing detected SMPS neutron star properties.
        dictionary_detected_HTRU_low_mid (dict): Dictionary containing detected HTRU low and mid-latitude neutron star properties.
        dictionary_detected_HTRU_high (dict): Dictionary containing detected HTRU high-latitude neutron star properties.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for
            detected neutron stars from PMPS, SMPS, HTRU low/mid and HTRU high.
    """

    # Generating two header lines and merging them using MultiIndex.
    parameters = [
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
    units = [
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

    # Parameters and units for HTRU_low_mid
    parameters_HTRU_low_mid = parameters + ["HTRU_low", "HTRU_mid"]
    units_HTRU_low_mid = units + ["", ""]

    # Build DataFrames
    df_PMPS = build_dataframe(dictionary_detected_PMPS, parameters, units)
    df_SMPS = build_dataframe(dictionary_detected_SMPS, parameters, units)
    df_HTRU_low_mid = build_dataframe(
        dictionary_detected_HTRU_low_mid,
        parameters_HTRU_low_mid,
        units_HTRU_low_mid,
    )
    df_HTRU_high = build_dataframe(
        dictionary_detected_HTRU_high, parameters, units
    )

    return df_PMPS, df_SMPS, df_HTRU_low_mid, df_HTRU_high


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
    prof_log_path = pathlib.Path().joinpath(output_path, cfg["profile_log"])
    prof_json_path = pathlib.Path().joinpath(output_path, cfg["profile_json"])

    # If already present, remove the profile.json and profile.log files to prevent
    # interrupted server connection issues.
    if os.path.exists(prof_json_path):
        os.remove(prof_json_path)

    if os.path.exists(prof_log_path):
        os.remove(prof_log_path)

    # Update the paths to profile.json and profile.log files in the configuration file.
    cfg["profile_json"] = str(prof_json_path)
    cfg["profile_log"] = str(prof_log_path)

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

    # Initialize the surveys.
    (
        radio_surveys,
        dictionary_detected_PMPS,
        dictionary_detected_SMPS,
        dictionary_detected_HTRU_low_mid,
        dictionary_detected_HTRU_high,
    ) = initialize_radio_surveys(cfg)

    n_detected_real_PMPS = cfg["detected_real_PMPS"]
    n_detected_real_SMPS = cfg["detected_real_SMPS"]
    n_detected_real_HTRU_low_mid = cfg["detected_real_htru_low_mid"]
    n_detected_real_HTRU_high = cfg["detected_real_htru_high"]

    # Import the maximum age value from the dynamical configuration file.
    # This is needed for the computation of the birth rate.
    with open(dyn_path_config, "r") as f:
        config_dyn = json.load(f)

    t_max = config_dyn["t_age_max"] / 100  # Maximum time in centuries.

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

        # These variables count how many stars we detect in total in each survey.
        n_detected_sim_PMPS = 0
        n_detected_sim_SMPS = 0
        n_detected_sim_HTRU_low_mid = 0
        n_detected_sim_HTRU_high = 0

        # In these variables we save how many stars we detect when we reach the desirable number in each survey.
        n_detected_sim_PMPS_at_match = 0
        n_detected_sim_SMPS_at_match = 0
        n_detected_sim_HTRU_low_mid_at_match = 0
        n_detected_sim_HTRU_high_at_match = 0

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

                # If the percentage of all surveys is over 90% reduce the batch size.
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

                # Load a chunk of dynamically evolved neutron stars from the dynamical database.
                database_dyn_chunk = load_database_dyn(
                    dyn_path_pop,
                    n_batchsize,
                    config_dyn["NS_number"],
                    idx_remove,
                )

                # Filter the loaded database chuck with the surveys sky coverage.
                database_coverage, idx_remove = apply_surveys_coverage(
                    radio_surveys,
                    database_dyn_chunk,
                    idx_remove,
                    dist_cutoff=30.0,
                )

                # Initialize neutron star population properties.
                pop_magrot_initial = initialize_population_magrot(
                    database_coverage
                )

            with timewith.TimeWith(
                "[SimulatePopulationDetection]",
                cfg["profile_log"],
                cfg["profile_json"],
                cfg["show_profiling"],
            ):

                # ===================== MAGNETO-ROTATIONAL EVOLUTION ========================

                # Determine the evolved magnetic field, misalignment angle and rotation period.
                pop_magrot_final = evolve_population_magrot(
                    cfg, pop_magrot_initial, database_coverage, output_path
                )

                pop_final = database_coverage | pop_magrot_final

                # ===================== RADIO DETECTION ========================

                pop_intercepted_radio = radio_intercepted(pop_final)

                if len(pop_intercepted_radio["age"]) == 0:
                    break

                (
                    update_dictionary_detected_PMPS,
                    update_dictionary_detected_SMPS,
                    update_dictionary_detected_HTRU_low_mid,
                    update_dictionary_detected_HTRU_high,
                ) = radio_detection(radio_surveys, pop_intercepted_radio)

                n_detected_sim_PMPS += len(
                    update_dictionary_detected_PMPS["age"]
                )
                log.info(
                    f"Total number of neutron stars detected by the Parkes Multibeam Survey: {n_detected_sim_PMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with PMPS is reached.
                # This is needed to compute the birth rate derived from the PMPS detections.
                if (n_detected_sim_PMPS >= n_detected_real_PMPS) & (
                    stop_PMPS is False
                ):
                    n_detected_sim_PMPS_at_match = n_detected_sim_PMPS
                    stop_PMPS = True
                    n_created_PMPS = n_created

                n_detected_sim_SMPS += len(
                    update_dictionary_detected_SMPS["age"]
                )
                log.info(
                    f"Total number of neutron stars detected by the Swinburne Parkes Multibeam Survey: {n_detected_sim_SMPS}"
                )
                # Store the value of created neutron stars once the number of detected pulsars with SMPS is reached.
                # This is needed to compute the birth rate derived from the SMPS detections.
                if (n_detected_sim_SMPS >= n_detected_real_SMPS) & (
                    stop_SMPS is False
                ):
                    n_detected_sim_SMPS_at_match = n_detected_sim_SMPS
                    stop_SMPS = True
                    n_created_SMPS = n_created

                n_detected_sim_HTRU_low_mid += len(
                    update_dictionary_detected_HTRU_low_mid["age"]
                )

                log.info(
                    f"Total number of neutron stars detected by the HTRU mid and low latitude surveys: {n_detected_sim_HTRU_low_mid}"
                )

                # Store the value of created neutron stars once the number of detected pulsars with HTRU is reached.
                # This is needed to compute the birth rate derived from the HTRU detections.
                if (
                    n_detected_sim_HTRU_low_mid >= n_detected_real_HTRU_low_mid
                ) & (stop_HTRU_low_mid is False):

                    n_detected_sim_HTRU_low_mid_at_match = (
                        n_detected_sim_HTRU_low_mid
                    )
                    stop_HTRU_low_mid = True
                    n_created_HTRU_low_mid = n_created

                n_detected_sim_HTRU_high += len(
                    update_dictionary_detected_HTRU_high["age"]
                )

                log.info(
                    f"Total number of neutron stars detected by the HTRU high latitude survey: {n_detected_sim_HTRU_high}"
                )

                # Store the value of created neutron stars once the number of detected pulsars with HTRU is reached.
                # This is needed to compute the birth rate derived from the HTRU detections.
                if (n_detected_sim_HTRU_high >= n_detected_real_HTRU_high) & (
                    stop_HTRU_high is False
                ):
                    n_detected_sim_HTRU_high_at_match = (
                        n_detected_sim_HTRU_high
                    )
                    stop_HTRU_high = True
                    n_created_HTRU_high = n_created

                # Update the dictionaries containing the detection information.
                dictionary_detected_PMPS = {
                    key: value + update_dictionary_detected_PMPS[key]
                    for key, value in dictionary_detected_PMPS.items()
                }

                dictionary_detected_SMPS = {
                    key: value + update_dictionary_detected_SMPS[key]
                    for key, value in dictionary_detected_SMPS.items()
                }

                dictionary_detected_HTRU_low_mid = {
                    key: value + update_dictionary_detected_HTRU_low_mid[key]
                    for key, value in dictionary_detected_HTRU_low_mid.items()
                }

                dictionary_detected_HTRU_high = {
                    key: value + update_dictionary_detected_HTRU_high[key]
                    for key, value in dictionary_detected_HTRU_high.items()
                }

                # ==========================================================

                # Remove from the dynamical database the stars that have been detected or
                # that are outside the sky coverage of the surveys.
                idx_det_PMPS = update_dictionary_detected_PMPS["idx"]
                idx_det_SMPS = update_dictionary_detected_SMPS["idx"]
                idx_det_HTRU_low_mid = update_dictionary_detected_HTRU_low_mid[
                    "idx"
                ]
                idx_det_HTRU_high = update_dictionary_detected_HTRU_high["idx"]

                idx_det_tot = list(
                    set().union(
                        idx_det_PMPS,
                        idx_det_SMPS,
                        idx_det_HTRU_low_mid,
                        idx_det_HTRU_high,
                    )
                )
                idx_remove += idx_det_tot

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
        cfg["birth_rate_PMPS_at_match"] = birth_rate_PMPS
        cfg["birth_rate_SMPS_at_match"] = birth_rate_SMPS
        cfg["birth_rate_HTRU_low_mid_at_match"] = birth_rate_HTRU_low_mid
        cfg["birth_rate_HTRU_high_at_match"] = birth_rate_HTRU_high

        # Add information on the number of detected neutron stars for each simulated survey at the point where
        # our simulation reaches the number of observed neutron stars specified for the real surveys.
        # Note: these numbers are not necessarily identical, because we batch our neutron star generation.
        cfg["n_detected_sim_PMPS_at_match"] = n_detected_sim_PMPS_at_match
        cfg["n_detected_sim_SMPS_at_match"] = n_detected_sim_SMPS_at_match
        cfg[
            "n_detected_sim_HTRU_low_mid_at_match"
        ] = n_detected_sim_HTRU_low_mid_at_match
        cfg[
            "n_detected_sim_HTRU_high_at_match"
        ] = n_detected_sim_HTRU_high_at_match

        # Add information on the number of detected star for each survey at the end of the simulation.
        cfg["n_detected_sim_PMPS_tot"] = n_detected_sim_PMPS
        cfg["n_detected_sim_SMPS_tot"] = n_detected_sim_SMPS
        cfg["n_detected_sim_HTRU_low_mid_tot"] = n_detected_sim_HTRU_low_mid
        cfg["n_detected_sim_HTRU_high_tot"] = n_detected_sim_HTRU_high

        # Add the parameters from the dynamical database to the configuration file.
        cfg["t_max"] = config_dyn["t_age_max"]
        cfg["NS_number"] = config_dyn["NS_number"]
        cfg["kick_model"] = config_dyn["kick_model"]
        cfg["sigma_k"] = config_dyn["sigma_k"]
        cfg["vk_c"] = config_dyn["vk_c"]
        cfg["h_c"] = config_dyn["h_c"]

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

            (
                df_PMPS,
                df_SMPS,
                df_HTRU_low_mid,
                df_HTRU_high,
            ) = create_output_dataframe(
                dictionary_detected_PMPS,
                dictionary_detected_SMPS,
                dictionary_detected_HTRU_low_mid,
                dictionary_detected_HTRU_high,
            )

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
                output_path, "survey_HTRU_low_mid_results.pkl.gz"
            )
            df_HTRU_low_mid.to_pickle(
                HTRU_output_low_mid_path, compression="gzip"
            )

            HTRU_high_output_path = pathlib.Path().joinpath(
                output_path, "survey_HTRU_high_results.pkl.gz"
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
                f"Output of the detected population with HTRU high survey generated in {os.getcwd()}/{HTRU_high_output_path}"
            )

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
