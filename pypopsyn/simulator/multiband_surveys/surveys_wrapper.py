"""
    A wrapper module to include all methods used in the script simulate_population_magrot_det.py to initialize and
    simulate surveys to detect neutron stars in different wavelengths.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""
import functools
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.multiband_surveys.survey_xray as sx
from pypopsyn.simulator.config_simulator import cfg


@dataclass
class SurveyData:
    """
    A dataclass object to store all the survey data for a simulation.

    Attributes:
        surveys_cfg (Dict): A dictionary to save all the survey information from config_simulator.
        surveys_radio (Dict): A dictionary to store the radio survey class objects.
        n_detected_sim (Dict): A dictionary to store the number of stars detected in each survey.
        n_detected_complete_sim (Dict): A dictionary to store how many neutron stars are detected in the flux ranges
            where we assume completeness.
        percentage_detected (Dict): A dictionary storing the percentage of detections compared to real detected numbers.
        n_created_at_match (Dict): A dictionary storing how many stars we have created to reach the desirable number in
            each survey.
        n_detected_sim_at_match (Dict): A dictionary storing how many stars we have detected when we reach the desirable
            number in each survey.
        batchsize_adjust_flags (Dict): A dictionary storing the flags for adjusting batch size as we approach the target
            detection number for all surveys..
        stop_flags (Dict): A dictionary storing the flags for stopping the simulation as we reach the target detection
            number for all surveys.
        dictionary_detected_radio (Dict): A dictionary storing all the properties of neutron stars detected in the radio
            surveys.
        dictionary_detected_xray (Optional[Dict]): A dictionary storing all the properties of neutron stars detected in
            the X-ray surveys if cfg["xray_simulation"] = True.
        surveys_xray (Optional[Dict]): A dictionary storing all the X-ray survey class objects if
            cfg["xray_simulation"] = True.
    """

    surveys_cfg: Dict
    surveys_radio: Dict
    n_detected_sim: Dict
    n_detected_complete_sim: Dict
    percentage_detected: Dict
    n_created_at_match: Dict
    n_detected_sim_at_match: Dict
    batchsize_adjust_flags: Dict
    stop_flags: Dict
    dictionary_detected_radio: Dict
    dictionary_detected_xray: Optional[Dict] = None
    surveys_xray: Optional[Dict] = None


def initialize_radio_surveys() -> Tuple[dict, dict]:
    """
    Initialize and return radio survey objects and empty detection dictionaries.

    Returns:
        (Tuple[dict, dict]): A tuple of dictionaries containing the following information:

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
        "tau_sc": [],
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


def initialize_xray_surveys() -> Tuple[dict, dict]:
    """
    Initialize and return X-ray survey detection dictionaries.

    Returns:
        (Tuple[dict, dict]): A tuple of dictionaries containing the following information:

            - A dictionary of initialized X-ray survey objects, keyed by survey names.
            - A dictionary for storing detected neutron star data for the X-ray survey.
    """
    # Get the path to the software directory.
    base_path = pathlib.Path(cfg["path_to_software"])

    # Initialize survey objects.
    xray_surveys = {}

    for survey_name, survey_dict in cfg["surveys_xray"].items():
        xray_surveys[survey_name] = sx.SurveyXray(
            str(base_path.joinpath(survey_dict["path"]))
        )

    # Detection dictionary template.
    detection_template = {
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
        "B_initial": [],
        "chi": [],
        "P": [],
        "P_dot": [],
        "L_x_therm": [],
        "S_x_rcs_abs": [],
        "S_x_bb_abs": [],
        "idx": [],
    }

    # Initialize detection dictionaries for each survey.
    detection_dictionaries = {
        survey_name: detection_template.copy()
        for survey_name in cfg["surveys_xray"].keys()
    }

    return xray_surveys, detection_dictionaries


def initialize_all_surveys() -> SurveyData:
    """
    Initialize all surveys.

    Returns:
        (SurveyData): A SurveyData object containing all survey data.
    """

    # Copy survey configuration data from the configuration file.
    surveys_cfg = cfg["surveys_radio"].copy()

    # Initialize radio surveys.
    surveys_radio, dictionary_detected_radio = initialize_radio_surveys()

    kwargs = {}

    if cfg["simulation_xray"]:
        surveys_cfg.update(cfg["surveys_xray"].copy())

        # Initialize X-ray surveys.
        surveys_xray, dictionary_detected_xray = initialize_xray_surveys()

        kwargs["surveys_xray"] = surveys_xray
        kwargs["dictionary_detected_xray"] = dictionary_detected_xray

    survey_data_class = SurveyData(
        surveys_cfg=surveys_cfg,
        surveys_radio=surveys_radio,
        n_detected_sim={survey: 0 for survey in surveys_cfg},
        n_detected_complete_sim={survey: 0 for survey in surveys_cfg},
        percentage_detected={survey: 0 for survey in surveys_cfg},
        n_created_at_match={survey: 0 for survey in surveys_cfg},
        n_detected_sim_at_match={survey: 0 for survey in surveys_cfg},
        batchsize_adjust_flags={0.9: False, 0.95: False},
        stop_flags={survey: False for survey in surveys_cfg},
        dictionary_detected_radio=dictionary_detected_radio,
        **kwargs,
    )

    return survey_data_class


def apply_surveys_coverage(
    surveys_radio: dict,
    surveys_xray: dict,
    dyn_database_dict: dict,
    idx_remove: list,
    dist_cutoff: float,
) -> Tuple[dict, list]:
    """
    Apply survey coverage criteria to filter a dynamic population dataset based on sky coverage of all surveys and a
    distance cutoff, and update the indices of entries to be removed.

    Args:
        surveys_radio (dict): A dictionary of radio survey objects, containing the information on the sky coverage.
        surveys_xray (dict): A dictionary of X-ray survey objects, containing the information on the sky coverage.
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

    survey_radio_names = list(surveys_radio.keys())
    survey_xray_names = []
    coverage_survey_radio = {}
    coverage_survey_xray = {}
    coverage_xray_tot = {}

    if surveys_xray is not None:
        survey_xray_names = list(surveys_xray.keys())
        coverage_survey_xray = {}

    for name in survey_radio_names:
        # Evaluate the sky coverage for each radio survey.
        coverage_survey_radio[name] = surveys_radio[name].sky_coverage(
            dyn_database_dict["ra"],
            dyn_database_dict["dec"],
            dyn_database_dict["l"],
            dyn_database_dict["b"],
        )

    # Combine the coverage masks of all selected radio surveys into a single mask.
    # It performs a logical OR (|) across all coverage arrays in coverage_survey_radio,
    # for each survey name in survey_radio_names.
    # The result is a single array where a position is True if it is covered by any survey.
    coverage_radio_tot = (
        functools.reduce(
            lambda a, b: a | b,
            (coverage_survey_radio[name] for name in survey_radio_names),
        )
    ) & dist_mask

    if surveys_xray is not None:
        # Evaluate the sky coverage for each X-ray survey.
        for name in survey_xray_names:
            coverage_survey_xray[name] = surveys_xray[name].sky_coverage(
                dyn_database_dict["ra"],
                dyn_database_dict["dec"],
                dyn_database_dict["l"],
                dyn_database_dict["b"],
            )

        # Combine the coverage masks of all selected X-ray surveys into a single mask.
        # It performs a logical OR (|) across all coverage arrays in coverage_survey_xray,
        # for each survey name in survey_xray_names.
        # The result is a single array where a position is True if it is covered by any survey.
        coverage_xray_tot = (
            functools.reduce(
                lambda a, b: a | b,
                (coverage_survey_xray[name] for name in survey_xray_names),
            )
        ) & dist_mask

    # Combine the total sky coverage for the radio and X-ray surveys together.
    if surveys_xray is not None:
        coverage_tot = coverage_radio_tot | coverage_xray_tot

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
        # Add the coverage data for each survey.
        coverage_key = f"coverage_radio_{survey_name}"
        dictionary_coverage_database[coverage_key] = coverage_survey_radio[
            survey_name
        ][coverage_tot]

    if surveys_xray is not None:
        dictionary_coverage_database["coverage_xray"] = coverage_xray_tot[
            coverage_tot
        ]

        for survey_name in survey_xray_names:
            # Add the coverage data for each survey.
            coverage_key = f"coverage_xray_{survey_name}"
            dictionary_coverage_database[coverage_key] = coverage_survey_xray[
                survey_name
            ][coverage_tot]

    # Remove stars that do not fall into the total sky coverage.
    idx = dyn_database_dict["idx"]
    out_coverage = np.invert(coverage_tot)
    idx_remove += idx[out_coverage].tolist()

    return dictionary_coverage_database, idx_remove


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
    # Initialize variables for HTRU_low and HTRU_mid.
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
        detected_dictionaries[survey_name] = update_filtered_dictionary(
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
        detected_dictionaries["HTRU_low_mid"] = update_filtered_dictionary(
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

        # Remove the dictionaries containing the results for the separated HTRU low and mid surveys.
        del detected_dictionaries["HTRU_low"]
        del detected_dictionaries["HTRU_mid"]

    return detected_dictionaries


def xray_detection(
    xray_surveys: dict,
    dict_xray_pop: dict,
) -> dict:
    """
    This function detects neutron stars by modelling some observational biases and updates their properties.

    Args:
        xray_surveys (dict): Dictionary containing the X-ray survey objects.
        dict_xray_pop (dict): Dictionary containing the properties of the evolved neutron star population.

    Returns:
        (dict): A dictionary containing the properties of neutron stars detected in X-rays.
    """
    # Process each survey.
    detected_dictionaries = {}
    for survey_name in xray_surveys:
        detected_mask = xray_surveys[survey_name].detected_xray_population(
            dict_xray_pop["S_x_rcs_abs"],
            dict_xray_pop["outburst"],
        )
        detected_dictionaries[survey_name] = update_filtered_dictionary(
            dict_xray_pop,
            detected_mask,
        )

    return detected_dictionaries


def update_filtered_dictionary(
    dict_to_update: dict, mask: np.ndarray, **kwargs: np.ndarray
) -> dict:
    """
    This function updates a dictionary to include for each key only the values corresponding to a given boolean mask.

    Args:
        dict_to_update (dict): The original dictionary containing properties of neutron stars.
        mask (np.ndarray): Boolean mask indicating which elements for each key in `dict_to_update` have to be included.
        **kwargs (np.ndarray): Additional property values provided as keyword arguments.
            These properties will also be filtered using the `mask`.

    Returns:
        (dict): A new dictionary containing only the filtered values from `dict_to_update` and combining
            the keys already present in the original dictionary with the ones provided in `kwargs`.
    """

    # Extract keys from the dictionary that has to be updated.
    properties = list(dict_to_update.keys())

    # Filtering the values in the original dictionary.
    filtered_dict = {
        prop: dict_to_update[prop][mask].tolist() for prop in properties
    }

    # Add and filter the properties specified in `kwargs` to another dictionary.
    additional_filtered_dict = {
        key: value[mask].tolist() for key, value in kwargs.items()
    }

    # Combine the two dictionaries.
    combined_dict = filtered_dict | additional_filtered_dict

    return combined_dict


def update_survey_data(
    survey_data_class: SurveyData,
    pop_detected_dict_update: dict,
    survey_type: str,
    n_created: int,
    idx_remove: list,
    logger: logging.Logger,
) -> None:
    """
    This function updates the survey data dictionaries to include only the properties of neutron stars that are
    detected in the surveys.

    Args:
        survey_data_class (SurveyData): The SurveyData dataclass containing the data of all neutron star surveys.
        pop_detected_dict_update (dict): A dictionary containing the properties of neutron stars that have been
            detected in the surveys.
        survey_type (str): A string specifying which survey to update, "Radio" or "X-ray".
        n_created (int): The total number of neutron stars created so far.
        idx_remove (list): A list of indices of the neutron stars that have been detected and have to be removed
            from the dynamical database.
        logger (logging.Logger): A logger instance to log messages.
    """

    surveys_cfg = survey_data_class.surveys_cfg
    n_detected_sim = survey_data_class.n_detected_sim
    n_detected_complete_sim = survey_data_class.n_detected_complete_sim
    stop_flags = survey_data_class.stop_flags
    n_detected_sim_at_match = survey_data_class.n_detected_sim_at_match
    n_created_at_match = survey_data_class.n_created_at_match
    dictionary_detected_radio = survey_data_class.dictionary_detected_radio
    dictionary_detected_xray = survey_data_class.dictionary_detected_xray

    for survey in pop_detected_dict_update:

        if survey_type == "radio":
            # Check the number of detected pulsars for each survey.
            n_detected_sim[survey] += len(
                pop_detected_dict_update[survey]["age"]
            )
            n_detected_complete_sim[survey] += len(
                pop_detected_dict_update[survey]["age"]
            )
            logger.info(
                f"Total number of neutron stars detected by {survey}: {n_detected_sim[survey]}"
            )
            # Since we assume that the radio surveys are complete, i.e.,
            # n_detected_complete_sim = n_detected_sim, if the number of simulated detected pulsars
            # matches the real one, store the value of created neutron stars.
            if (
                n_detected_complete_sim[survey]
                >= surveys_cfg[survey]["detected_real"]
                and not stop_flags[survey]
            ):
                n_detected_sim_at_match[survey] = n_detected_sim[survey]
                stop_flags[survey] = True
                n_created_at_match[survey] = n_created

            # Update the detection dictionaries.
            dictionary_detected_radio[survey] = {
                key: value + pop_detected_dict_update[survey][key]
                for key, value in dictionary_detected_radio[survey].items()
            }
            # Update the index list to remove the stars that have been detected from the dynamical database.
            idx_det = list(dictionary_detected_radio[survey]["idx"])
            idx_remove += idx_det

        elif survey_type == "X-ray":
            # For the X-ray survey we are not complete and we do not control well the observational biases. Therefore
            # we consider a flux threshold above which we assume we are complete and try to match the number of observed
            # sources above this flux threshold. See the config_simulator file for more details.
            n_detected_sim[survey] += len(
                pop_detected_dict_update[survey]["age"]
            )
            mask_completeness = (
                np.array(pop_detected_dict_update[survey]["S_x_rcs_abs"])
                > surveys_cfg[survey]["flux_threshold_completeness"]
            )
            n_detected_complete_sim[survey] += len(
                np.array(pop_detected_dict_update[survey]["age"])[
                    mask_completeness
                ]
            )
            logger.info(
                f"Total number of neutron stars detected by {survey}: {n_detected_sim[survey]} (above completeness flux threshold: {n_detected_complete_sim[survey]})"
            )
            # If the number of simulated detected pulsars above the completeness flux threshold matches the
            # real one, store the value of created neutron stars.
            if (
                n_detected_complete_sim[survey]
                >= surveys_cfg[survey]["detected_real"]
                and not stop_flags[survey]
            ):
                n_detected_sim_at_match[survey] = n_detected_sim[survey]
                stop_flags[survey] = True
                n_created_at_match[survey] = n_created
            dictionary_detected_xray[survey] = {
                key: value + pop_detected_dict_update[survey][key]
                for key, value in dictionary_detected_xray[survey].items()
            }
            # Update the index list to remove the stars that have been detected from the dynamical database.
            idx_det = list(dictionary_detected_xray[survey]["idx"])
            idx_remove += idx_det

        else:
            logger.error(f"Unknown survey type: {survey_type}")
            sys.exit(1)


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
    dictionary_detected_xray: dict,
) -> dict:
    """
    Creates Pandas DataFrames containing the information on the detected neutron stars for each survey.

    Args:
        dictionary_detected_radio (dict): Dictionary containing detected neutron star properties for each radio survey.
        dictionary_detected_xray (dict): Dictionary containing detected neutron star properties for an X-ray survey.

    Returns:
        (dict): A dictionary of DataFrames, one for each survey containing detected neutron stars' information.
    """

    # Initialize an empty dictionary to store the resulting DataFrames.
    dfs = {}

    # Defining the parameters and units that are common for all radio surveys.
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
        "tau_sc",
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
        "[s]",
        "",
    ]

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

    if dictionary_detected_xray is not None:
        parameters_xray = [
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
            "B_initial",
            "B",
            "chi",
            "P",
            "P_dot",
            "L_x_therm",
            "S_x_rcs_abs",
            "S_x_bb_abs",
        ]
        units_xray = [
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
            "[G]",
            "[rad]",
            "[s]",
            "[s s^-1]",
            "[erg s^-1]",
            "[erg s^-1 cm^-2]",
            "[erg s^-1 cm^-2]",
        ]

        # Loop over each survey's detected dictionary and generate the corresponding DataFrame.
        for survey_name, survey_data in dictionary_detected_xray.items():
            # Build the DataFrame using the appropriate parameters and units.
            df = build_dataframe(survey_data, parameters_xray, units_xray)
            dfs[
                survey_name
            ] = df  # Store the DataFrame in the dictionary with survey_name as key.

    return dfs


def adjust_n_batchsize(survey_data_class) -> int:
    """
    To speed up the simulation, generate new neutron stars in batches.
    The batchsize is adjusted as the synthetic population approaches the observed number of neutron stars in the real surveys.
    This guarantees a better fine tuning of the simulated detected numbers.

    Args:
        survey_data_class (SurveyData): The SurveyData dataclass containing the data of all neutron star surveys.
    """

    surveys_cfg = survey_data_class.surveys_cfg
    n_detected_complete_sim = survey_data_class.n_detected_complete_sim
    percentage_detected = survey_data_class.percentage_detected
    batchsize_adjusting_flags = survey_data_class.batchsize_adjust_flags

    # Evaluate the percentage of neutron stars detected by the simulated
    # surveys with respect to the real surveys and adjust the batch size accordingly.
    for survey in surveys_cfg:
        percentage_detected[survey] = (
            n_detected_complete_sim[survey]
            / surveys_cfg[survey]["detected_real"]
        )

    if (
        all(value > 0.9 for value in percentage_detected.values())
        and not batchsize_adjusting_flags[0.9]
    ):
        batchsize_adjusting_flags[0.9] = True
        n_batchsize = 10000
    elif (
        all(value > 0.95 for value in percentage_detected.values())
        and not batchsize_adjusting_flags[0.95]
    ):
        batchsize_adjusting_flags[0.95] = True
        n_batchsize = 5000

    else:
        n_batchsize = 100000

    return n_batchsize
