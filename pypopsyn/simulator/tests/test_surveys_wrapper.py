"""
    Tests for the surveys_wrapper module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.multiband_surveys.survey_xray as sx
import pypopsyn.simulator.multiband_surveys.surveys_wrapper as sw
from pypopsyn.simulator.config_simulator import cfg

# Set some radio surveys in the configuration file for testing purposes.
cfg["surveys_radio"]: dict = {
    "PMPS": {
        "path": "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
        "detected_real": 1045,
    },
    "HTRU_low_mid": {
        "path_low": "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
        "path_mid": "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
        "detected_real": 1037,
    },
}

cfg["surveys_xray"]: dict = {
    "xray_flux_threshold": {
        "path": "pypopsyn/simulator/multiband_surveys/xray_flux_threshold_parameters.json",
        "detected_real": 31,
    },
    "xray_realistic": {
        "path": "pypopsyn/simulator/multiband_surveys/xray_realistic_parameters.json",
        "detected_real": 31,
    },
}


class MockSurveyRadio:
    """Mock class to simulate a radio survey."""

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, MockSurveyRadio) and self.name == other.name

    def sky_coverage(self, ra, dec, l_gal, b_gal):
        """Mock implementation of sky_coverage."""
        # For simplicity, return a mask that includes only the first half of the stars.
        return np.array(
            [True if i % 2 == 0 else False for i in range(len(ra))]
        )

    def detected_radio_population(
        self,
        w_int_s,
        DM,
        P,
        age,
        coverage,
        l_gal,
        b_gal,
        S_radio_bol,
        spectral_index,
        tau_sc,
    ):
        """Mock implementation of detected_radio_population."""
        # For simplicity, return the same output for all the surveys.
        return (
            np.array([True, False]),  # detected_radio
            np.array([0.1, 0.2]),  # w_eff
            np.array([0.001, 0.002]),  # S_radio_obs_mean
            np.array([0.003, 0.004]),  # S_radio_obs_mean_1400
        )


class MockSurveyXray:
    """Mock class to simulate a X-ray survey."""

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, MockSurveyXray) and self.name == other.name

    def sky_coverage(self, ra, dec, l_gal, b_gal):
        """Mock implementation of sky_coverage."""
        # For simplicity, return a mask that includes only the first half of the stars.
        return np.array(
            [True if i % 2 == 0 else False for i in range(len(ra))]
        )

    def detected_xray_population(self, S_x, outburst_mask):
        """Mock implementation of detected_xray_population."""
        # For simplicity, return the same output for all the surveys.
        return (np.array([True, False]),)  # detected_radio


@pytest.fixture()
def test_case_1():
    data = {
        "radio_surveys_expected": {
            "PMPS": MockSurveyRadio("PMPS"),
            "HTRU_low": MockSurveyRadio("HTRU_low"),
            "HTRU_mid": MockSurveyRadio("HTRU_mid"),
        },
        "detection_dicts_expected": {
            "PMPS": {
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
            },
            "HTRU_low_mid": {
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
                "HTRU_low": [],
                "HTRU_mid": [],
            },
        },
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "xray_surveys_expected": {
            "xray_flux_threshold": MockSurveyXray("xray_flux_threshold"),
            "xray_realistic": MockSurveyXray("xray_realistic"),
        },
        "detection_dicts_expected": {
            "xray_flux_threshold": {
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
            },
            "xray_realistic": {
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
            },
        },
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "SurveyData_without_xray_expected": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            surveys_xray=None,
            n_detected_sim={survey: 0 for survey in ("PMPS", "HTRU_low_mid")},
            percentage_detected={
                survey: 0 for survey in ("PMPS", "HTRU_low_mid")
            },
            n_created_at_match={
                survey: 0 for survey in ("PMPS", "HTRU_low_mid")
            },
            n_detected_sim_at_match={
                survey: 0 for survey in ("PMPS", "HTRU_low_mid")
            },
            batchsize_adjust_flags={0.9: False, 0.95: False},
            stop_flags={survey: False for survey in ("PMPS", "HTRU_low_mid")},
            dictionary_detected_radio={
                "PMPS": {
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
                },
                "HTRU_low_mid": {
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
                    "HTRU_low": [],
                    "HTRU_mid": [],
                },
            },
            dictionary_detected_xray=None,
        ),
        "SurveyData_with_xray_expected": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
                "xray_flux_threshold": {
                    "path": "pypopsyn/simulator/multiband_surveys/xray_flux_threshold_parameters.json",
                    "detected_real": 31,
                },
                "xray_realistic": {
                    "path": "pypopsyn/simulator/multiband_surveys/xray_realistic_parameters.json",
                    "detected_real": 31,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            surveys_xray={
                "xray_flux_threshold": MockSurveyXray("xray_flux_threshold"),
                "xray_realistic": MockSurveyXray("xray_realistic"),
            },
            n_detected_sim={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            percentage_detected={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            n_created_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            n_detected_sim_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            batchsize_adjust_flags={0.9: False, 0.95: False},
            stop_flags={
                survey: False
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            dictionary_detected_radio={
                "PMPS": {
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
                },
                "HTRU_low_mid": {
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
                    "HTRU_low": [],
                    "HTRU_mid": [],
                },
            },
            dictionary_detected_xray={
                "xray_flux_threshold": {
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
                },
                "xray_realistic": {
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
                },
            },
        ),
    }
    return data


@pytest.fixture()
def test_case_4():
    data = {
        "radio_surveys": {
            "PMPS": MockSurveyRadio("PMPS"),
            "HTRU_low": MockSurveyRadio("HTRU_low"),
            "HTRU_mid": MockSurveyRadio("HTRU_mid"),
        },
        "xray_surveys": {
            "xray_flux_threshold": MockSurveyXray("xray_flux_threshold"),
            "xray_realistic": MockSurveyXray("xray_realistic"),
        },
        "dyn_database_dict": {
            "age": np.array([1e6, 2e6]),
            "ra": np.array([50.0, 250.0]),
            "dec": np.array([-50.0, 50.0]),
            "l": np.array([-50.0, 50.0]),
            "b": np.array([-20.0, 10.0]),
            "dist": np.array([2.0, 10.0]),
            "pm_ra": np.array([-50.0, 50.0]),
            "pm_dec": np.array([-50.0, 50.0]),
            "v_ls": np.array([-50.0, 50.0]),
            "idx": np.array([0, 1]),
        },
        "idx_remove": [],
        "dist_cutoff": 5.0,
        "expected_keys_with_xray": {
            "age",
            "ra",
            "dec",
            "l",
            "b",
            "dist",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "idx",
            "coverage_radio",
            "coverage_radio_PMPS",
            "coverage_radio_HTRU_low",
            "coverage_radio_HTRU_mid",
            "coverage_xray",
            "coverage_xray_xray_flux_threshold",
            "coverage_xray_xray_realistic",
        },
        "expected_keys_without_xray": {
            "age",
            "ra",
            "dec",
            "l",
            "b",
            "dist",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "idx",
            "coverage_radio",
            "coverage_radio_PMPS",
            "coverage_radio_HTRU_low",
            "coverage_radio_HTRU_mid",
        },
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "radio_surveys": {
            "PMPS": MockSurveyRadio("PMPS"),
            "HTRU_low": MockSurveyRadio("HTRU_low"),
            "HTRU_mid": MockSurveyRadio("HTRU_mid"),
        },
        "dictionary_intercepted_radio": {
            "age": np.array([1e6, 2e6]),
            "ra": np.array([180.0, 190.0]),
            "dec": np.array([-45.0, -50.0]),
            "l": np.array([120.0, 130.0]),
            "b": np.array([10.0, 15.0]),
            "dist": np.array([2.0, 3.0]),
            "pm_ra": np.array([10.0, 20.0]),
            "pm_dec": np.array([-5.0, -10.0]),
            "v_ls": np.array([50.0, 60.0]),
            "B": np.array([1e12, 2e12]),
            "chi": np.array([30.0, 40.0]),
            "P": np.array([0.5, 1.0]),
            "P_dot": np.array([1e-15, 2e-15]),
            "w_int": np.array([0.05, 0.1]),
            "DM": np.array([100.0, 200.0]),
            "idx": np.array([0, 1]),
            "L_radio_bol": np.array([1e30, 1e31]),
            "S_radio_bol": np.array([0.01, 0.02]),
            "spectral_index": np.array([-1.8, -1.8]),
            "tau_sc": np.array([0.001, 0.002]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
        },
        "expected_update_dictionary_detected": {
            "PMPS": {
                "age": np.array([1e6]),
                "ra": np.array([180.0]),
                "dec": np.array([-45.0]),
                "l": np.array([120.0]),
                "b": np.array([10.0]),
                "DM": np.array([100.0]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([10.0]),
                "pm_dec": np.array([-5.0]),
                "v_ls": np.array([50.0]),
                "B": np.array([1e12]),
                "chi": np.array([30.0]),
                "P": np.array([0.5]),
                "P_dot": np.array([1e-15]),
                "L_radio_bol": np.array([1e30]),
                "S_radio_bol": np.array([0.01]),
                "S_radio_obs_mean": np.array([0.001]),
                "S_radio_obs_mean_1400": np.array([0.003]),
                "w_int": np.array([0.05]),
                "w_eff": np.array([0.1]),
                "spectral_index": np.array([-1.8]),
                "tau_sc": np.array([0.001]),
                "coverage_radio_PMPS": np.array([True]),
                "coverage_radio_HTRU_low": np.array([True]),
                "coverage_radio_HTRU_mid": np.array([True]),
                "idx": np.array([0]),
            },
            "HTRU_low_mid": {
                "age": np.array([1e6]),
                "ra": np.array([180.0]),
                "dec": np.array([-45.0]),
                "l": np.array([120.0]),
                "b": np.array([10.0]),
                "DM": np.array([100.0]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([10.0]),
                "pm_dec": np.array([-5.0]),
                "v_ls": np.array([50.0]),
                "B": np.array([1e12]),
                "chi": np.array([30.0]),
                "P": np.array([0.5]),
                "P_dot": np.array([1e-15]),
                "L_radio_bol": np.array([1e30]),
                "S_radio_bol": np.array([0.01]),
                "S_radio_obs_mean": np.array([0.001]),
                "S_radio_obs_mean_1400": np.array([0.003]),
                "w_int": np.array([0.05]),
                "w_eff": np.array([0.1]),
                "spectral_index": np.array([-1.8]),
                "tau_sc": np.array([0.001]),
                "coverage_radio_PMPS": np.array([True]),
                "coverage_radio_HTRU_low": np.array([True]),
                "coverage_radio_HTRU_mid": np.array([True]),
                "idx": np.array([0]),
                "HTRU_low": np.array([True]),
                "HTRU_mid": np.array([True]),
            },
        },
    }

    return data


@pytest.fixture()
def test_case_6():
    data = {
        "xray_surveys": {
            "xray_flux_threshold": MockSurveyXray("xray_flux_threshold"),
            "xray_realistic": MockSurveyXray("xray_realistic"),
        },
        "dictionary_xray_pop": {
            "age": np.array([1e6, 2e6]),
            "l": np.array([-50.0, 50.0]),
            "b": np.array([-20.0, 10.0]),
            "ra": np.array([50.0, 250.0]),
            "dec": np.array([-50.0, 50.0]),
            "dist": np.array([2.0, 10.0]),
            "pm_ra": np.array([-50.0, 50.0]),
            "pm_dec": np.array([-50.0, 50.0]),
            "v_ls": np.array([-50.0, 50.0]),
            "P": np.array([0.01, 0.5]),
            "P_dot": np.array([1.0e-11, 1.0e-12]),
            "B_initial": np.array([1e12, 1e14]),
            "B": np.array([1e12, 1e14]),
            "chi": np.array([1.0, 2.0]),
            "L_x_therm": np.array([1e33, 1e34]),
            "S_x_rcs_abs": np.array([3.0e-12, 4.0e-16]),
            "S_x_bb_abs": np.array([2.0e-12, 3.0e-16]),
            "N_H": np.array([2.0e-21, 3.0e-21]),
            "idx": np.array([0, 1]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
            "coverage_radio": np.array([True, False]),
            "coverage_xray_xray_flux_threshold": np.array([True, True]),
            "coverage_xray_xray_realistic": np.array([True, True]),
            "coverage_xray": np.array([True, True]),
            "outburst": np.array([True, False]),
        },
        "update_dictionary_detected_x_expected": {
            "xray_flux_threshold": {
                "age": np.array([1e4]),
                "ra": np.array([50.0]),
                "dec": np.array([-50.0]),
                "l": np.array([-50.0]),
                "b": np.array([-20.0]),
                "N_H": np.array([2.0e-21]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([-50.0]),
                "pm_dec": np.array([-50.0]),
                "v_ls": np.array([-50.0]),
                "B_initial": np.array([1e12]),
                "B": np.array([1e12]),
                "chi": np.array([1.0]),
                "P": np.array([0.01]),
                "P_dot": np.array([1.0e-11]),
                "L_x_therm": np.array([1.0e34]),
                "S_x_rcs_abs": np.array([3.0e-12]),
                "S_x_bb_abs": np.array([2.0e-12]),
                "idx": np.array([0]),
                "coverage_radio": np.array([False]),
                "coverage_radio_HTRU_low": np.array([False]),
                "coverage_radio_HTRU_mid": np.array([False]),
                "coverage_radio_PMPS": np.array([False]),
                "coverage_xray_xray_flux_threshold": np.array([True]),
                "coverage_xray_xray_realistic": np.array([True]),
                "coverage_xray": np.array([True]),
                "outburst": np.array([True]),
            },
            "xray_realistic": {
                "age": np.array([1e4]),
                "ra": np.array([50.0]),
                "dec": np.array([-50.0]),
                "l": np.array([-50.0]),
                "b": np.array([-20.0]),
                "N_H": np.array([2.0e-21]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([-50.0]),
                "pm_dec": np.array([-50.0]),
                "v_ls": np.array([-50.0]),
                "B_initial": np.array([1e12]),
                "B": np.array([1e12]),
                "chi": np.array([1.0]),
                "P": np.array([0.01]),
                "P_dot": np.array([1.0e-11]),
                "L_x_therm": np.array([1.0e34]),
                "S_x_rcs_abs": np.array([3.0e-12]),
                "S_x_bb_abs": np.array([2.0e-12]),
                "idx": np.array([0]),
                "coverage_radio": np.array([False]),
                "coverage_radio_HTRU_low": np.array([False]),
                "coverage_radio_HTRU_mid": np.array([False]),
                "coverage_radio_PMPS": np.array([False]),
                "coverage_xray_xray_flux_threshold": np.array([True]),
                "coverage_xray_xray_realistic": np.array([True]),
                "coverage_xray": np.array([True]),
                "outburst": np.array([True]),
            },
        },
    }

    return data


@pytest.fixture()
def test_case_7():
    data = {
        "dict_to_update": {
            "age": np.array([100, 1.0e3, 1.0e5, 1.0e4]),
            "l": np.array([10.0, 11.0, 9.5, 12.0]),
        },
        "detected_mask": np.array([True, False, True, False]),
        "additional_properties": {
            "w_eff": np.array([0.1, 0.2, 0.3, 0.1]),
            "S_radio_obs_mean": np.array([1.0e-3, 1.0e-5, 0.1, 1.0e-4]),
        },
        "expected_output": {
            "age": [100, 1.0e5],
            "l": [10.0, 9.5],
            "w_eff": [0.1, 0.3],
            "S_radio_obs_mean": [1.0e-3, 0.1],
        },
    }

    return data


@pytest.fixture()
def test_case_8():
    data = {
        "SurveyData": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
                "xray_flux_threshold": {
                    "path": "pypopsyn/simulator/multiband_surveys/xray_flux_threshold_parameters.json",
                    "detected_real": 31,
                },
                "xray_realistic": {
                    "path": "pypopsyn/simulator/multiband_surveys/xray_realistic_parameters.json",
                    "detected_real": 31,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            surveys_xray={
                "xray_flux_threshold": MockSurveyXray("xray_flux_threshold"),
                "xray_realistic": MockSurveyXray("xray_realistic"),
            },
            n_detected_sim={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            percentage_detected={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            n_created_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            n_detected_sim_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            batchsize_adjust_flags={0.9: False, 0.95: False},
            stop_flags={
                survey: False
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                    "xray_flux_threshold",
                    "xray_realistic",
                )
            },
            dictionary_detected_radio={
                "PMPS": {
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
                },
                "HTRU_low_mid": {
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
                    "tau_sc": [],
                    "idx": [],
                    "HTRU_low": [],
                    "HTRU_mid": [],
                },
            },
            dictionary_detected_xray={
                "xray_flux_threshold": {
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
                    "B_initial": [],
                    "B": [],
                    "chi": [],
                    "P": [],
                    "P_dot": [],
                    "L_x_therm": [],
                    "S_x_rcs_abs": [],
                    "S_x_bb_abs": [],
                    "idx": [],
                },
                "xray_realistic": {
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
                    "B_initial": [],
                    "B": [],
                    "chi": [],
                    "P": [],
                    "P_dot": [],
                    "L_x_therm": [],
                    "S_x_rcs_abs": [],
                    "S_x_bb_abs": [],
                    "idx": [],
                },
            },
        ),
        "update_dictionary_detected_radio": {
            "PMPS": {
                "age": [1e6],
                "ra": [180.0],
                "dec": [-45.0],
                "l": [120.0],
                "b": [10.0],
                "DM": [100.0],
                "dist": [2.0],
                "pm_ra": [10.0],
                "pm_dec": [-5.0],
                "v_ls": [50.0],
                "B": [1e12],
                "chi": [30.0],
                "P": [0.5],
                "P_dot": [1e-15],
                "L_radio_bol": [1e30],
                "S_radio_bol": [0.01],
                "S_radio_obs_mean": [0.001],
                "S_radio_obs_mean_1400": [0.003],
                "w_int": [0.05],
                "w_eff": [0.1],
                "spectral_index": [-1.8],
                "tau_sc": [0.001],
                "coverage_radio_PMPS": [True],
                "coverage_radio_HTRU_low": [True],
                "coverage_radio_HTRU_mid": [True],
                "idx": [0],
            },
            "HTRU_low_mid": {
                "age": [1e6],
                "ra": [180.0],
                "dec": [-45.0],
                "l": [120.0],
                "b": [10.0],
                "DM": [100.0],
                "dist": [2.0],
                "pm_ra": [10.0],
                "pm_dec": [-5.0],
                "v_ls": [50.0],
                "B": [1e12],
                "chi": [30.0],
                "P": [0.5],
                "P_dot": [1e-15],
                "L_radio_bol": [1e30],
                "S_radio_bol": [0.01],
                "S_radio_obs_mean": [0.001],
                "S_radio_obs_mean_1400": [0.003],
                "w_int": [0.05],
                "w_eff": [0.1],
                "spectral_index": [-1.8],
                "tau_sc": [0.001],
                "coverage_radio_PMPS": [True],
                "coverage_radio_HTRU_low": [True],
                "coverage_radio_HTRU_mid": [True],
                "idx": [0],
                "HTRU_low": [True],
                "HTRU_mid": [True],
            },
        },
        "dictionary_detected_radio_expected": {
            "PMPS": {
                "age": [1e6],
                "ra": [180.0],
                "dec": [-45.0],
                "l": [120.0],
                "b": [10.0],
                "DM": [100.0],
                "dist": [2.0],
                "pm_ra": [10.0],
                "pm_dec": [-5.0],
                "v_ls": [50.0],
                "B": [1e12],
                "chi": [30.0],
                "P": [0.5],
                "P_dot": [1e-15],
                "L_radio_bol": [1e30],
                "S_radio_obs_mean": [0.001],
                "S_radio_obs_mean_1400": [0.003],
                "w_int": [0.05],
                "w_eff": [0.1],
                "spectral_index": [-1.8],
                "tau_sc": [0.001],
                "idx": [0],
            },
            "HTRU_low_mid": {
                "age": [1e6],
                "ra": [180.0],
                "dec": [-45.0],
                "l": [120.0],
                "b": [10.0],
                "DM": [100.0],
                "dist": [2.0],
                "pm_ra": [10.0],
                "pm_dec": [-5.0],
                "v_ls": [50.0],
                "B": [1e12],
                "chi": [30.0],
                "P": [0.5],
                "P_dot": [1e-15],
                "L_radio_bol": [1e30],
                "S_radio_obs_mean": [0.001],
                "S_radio_obs_mean_1400": [0.003],
                "w_int": [0.05],
                "w_eff": [0.1],
                "spectral_index": [-1.8],
                "tau_sc": [0.001],
                "idx": [0],
                "HTRU_low": [True],
                "HTRU_mid": [True],
            },
        },
        "n_created_expected": 1000,
        "idx_remove_expected": [0, 0],
        "update_dictionary_detected_xray": {
            "xray_flux_threshold": {
                "age": [1e4],
                "ra": [50.0],
                "dec": [-50.0],
                "l": [-50.0],
                "b": [-20.0],
                "N_H": [2.0e-21],
                "dist": [2.0],
                "pm_ra": [-50.0],
                "pm_dec": [-50.0],
                "v_ls": [-50.0],
                "B_initial": [1e12],
                "B": [1e12],
                "chi": [1.0],
                "P": [0.01],
                "P_dot": [1.0e-11],
                "L_x_therm": [1.0e34],
                "S_x_rcs_abs": [3.0e-12],
                "S_x_bb_abs": [2.0e-12],
                "idx": [0],
                "coverage_radio": [False],
                "coverage_radio_HTRU_low": [False],
                "coverage_radio_HTRU_mid": [False],
                "coverage_radio_PMPS": [False],
                "coverage_xray_xray_flux_threshold": [True],
                "coverage_xray_xray_realistic": [True],
                "coverage_xray": [True],
                "outburst": [True],
            },
            "xray_realistic": {
                "age": [1e4],
                "ra": [50.0],
                "dec": [-50.0],
                "l": [-50.0],
                "b": [-20.0],
                "N_H": [2.0e-21],
                "dist": [2.0],
                "pm_ra": [-50.0],
                "pm_dec": [-50.0],
                "v_ls": [-50.0],
                "B_initial": [1e12],
                "B": [1e12],
                "chi": [1.0],
                "P": [0.01],
                "P_dot": [1.0e-11],
                "L_x_therm": [1.0e34],
                "S_x_rcs_abs": [3.0e-12],
                "S_x_bb_abs": [2.0e-12],
                "idx": [0],
                "coverage_radio": [False],
                "coverage_radio_HTRU_low": [False],
                "coverage_radio_HTRU_mid": [False],
                "coverage_radio_PMPS": [False],
                "coverage_xray_xray_flux_threshold": [True],
                "coverage_xray_xray_realistic": [True],
                "coverage_xray": [True],
                "outburst": [True],
            },
        },
        "dictionary_detected_xray_expected": {
            "xray_flux_threshold": {
                "age": [1e4],
                "ra": [50.0],
                "dec": [-50.0],
                "l": [-50.0],
                "b": [-20.0],
                "N_H": [2.0e-21],
                "dist": [2.0],
                "pm_ra": [-50.0],
                "pm_dec": [-50.0],
                "v_ls": [-50.0],
                "B_initial": [1e12],
                "B": [1e12],
                "chi": [1.0],
                "P": [0.01],
                "P_dot": [1.0e-11],
                "L_x_therm": [1.0e34],
                "S_x_rcs_abs": [3.0e-12],
                "S_x_bb_abs": [2.0e-12],
                "idx": [0],
            },
            "xray_realistic": {
                "age": [1e4],
                "ra": [50.0],
                "dec": [-50.0],
                "l": [-50.0],
                "b": [-20.0],
                "N_H": [2.0e-21],
                "dist": [2.0],
                "pm_ra": [-50.0],
                "pm_dec": [-50.0],
                "v_ls": [-50.0],
                "B_initial": [1e12],
                "B": [1e12],
                "chi": [1.0],
                "P": [0.01],
                "P_dot": [1.0e-11],
                "L_x_therm": [1.0e34],
                "S_x_rcs_abs": [3.0e-12],
                "S_x_bb_abs": [2.0e-12],
                "idx": [0],
            },
        },
    }

    return data


@pytest.fixture()
def test_case_9():
    data = {
        "data_dict": {
            "mass": [1.4, 1.3],
            "radius": [10.0, 9.5],
        },
        "parameters": ["mass", "radius"],
        "units": ["[Msun]", "[kpc]"],
        "expected_columns": pd.MultiIndex.from_arrays(
            [
                ["mass", "radius"],
                ["[Msun]", "[kpc]"],
            ]
        ),
        "expected_data": {
            ("mass", "[Msun]"): [1.4, 1.3],
            ("radius", "[kpc]"): [10.0, 9.5],
        },
        "expected_df": pd.DataFrame(
            {
                ("mass", "[Msun]"): [1.4, 1.3],
                ("radius", "[kpc]"): [10.0, 9.5],
            }
        ),
    }

    return data


@pytest.fixture()
def test_case_10():
    data = {
        "dictionary_detected_radio": {
            "PMPS": {
                "age": [1e6, 2e6],
                "RA": [180.0, 190.0],
                "DEC": [45.0, 50.0],
                "l": [120.0, 130.0],
                "b": [30.0, 35.0],
                "DM": [10.0, 12.0],
                "d": [1.0, 1.5],
                "pm_RA": [3.0, 4.0],
                "pm_DEC": [2.0, 2.5],
                "v_ls": [100.0, 110.0],
                "B": [1e12, 1.1e12],
                "chi": [0.1, 0.2],
                "P": [0.5, 0.6],
                "P_dot": [1e-15, 1.1e-15],
                "L_radio_bol": [1e30, 1.1e30],
                "S_radio_obs_mean": [0.01, 0.02],
                "S_radio_obs_mean_1400": [0.005, 0.007],
                "w_int": [0.002, 0.003],
                "w_eff": [0.004, 0.005],
                "tau_sc": [0.001, 0.001],
                "spectral_index": [-1.4, -1.5],
            },
            "HTRU_low_mid": {
                "age": [3e6, 4e6],
                "RA": [200.0, 210.0],
                "DEC": [55.0, 60.0],
                "l": [140.0, 150.0],
                "b": [40.0, 45.0],
                "DM": [14.0, 15.0],
                "d": [2.0, 2.5],
                "pm_RA": [5.0, 6.0],
                "pm_DEC": [3.0, 3.5],
                "v_ls": [120.0, 130.0],
                "B": [1.2e12, 1.3e12],
                "chi": [0.3, 0.4],
                "P": [0.7, 0.8],
                "P_dot": [1.2e-15, 1.3e-15],
                "L_radio_bol": [1.2e30, 1.3e30],
                "S_radio_obs_mean": [0.03, 0.04],
                "S_radio_obs_mean_1400": [0.008, 0.009],
                "w_int": [0.006, 0.007],
                "w_eff": [0.008, 0.009],
                "tau_sc": [0.001, 0.001],
                "spectral_index": [-1.6, -1.7],
                ("HTRU_low", ""): [True, False],
                ("HTRU_mid", ""): [True, True],
            },
        },
        "dictionary_detected_xray": {
            "xray_flux_threshold": {
                "age": np.array([1e4]),
                "ra": np.array([50.0]),
                "dec": np.array([-50.0]),
                "l": np.array([-50.0]),
                "b": np.array([-20.0]),
                "N_H": np.array([2.0e-21]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([-50.0]),
                "pm_dec": np.array([-50.0]),
                "v_ls": np.array([-50.0]),
                "B_initial": np.array([1e12]),
                "B": np.array([1e12]),
                "chi": np.array([1.0]),
                "P": np.array([0.01]),
                "P_dot": np.array([1.0e-11]),
                "L_x_therm": np.array([1.0e34]),
                "S_x_rcs_abs": np.array([3.0e-12]),
                "S_x_bb_abs": np.array([2.0e-12]),
                "idx": np.array([0]),
            },
            "xray_realistic": {
                "age": np.array([1e4]),
                "ra": np.array([50.0]),
                "dec": np.array([-50.0]),
                "l": np.array([-50.0]),
                "b": np.array([-20.0]),
                "N_H": np.array([2.0e-21]),
                "dist": np.array([2.0]),
                "pm_ra": np.array([-50.0]),
                "pm_dec": np.array([-50.0]),
                "v_ls": np.array([-50.0]),
                "B_initial": np.array([1e12]),
                "B": np.array([1e12]),
                "chi": np.array([1.0]),
                "P": np.array([0.01]),
                "P_dot": np.array([1.0e-11]),
                "L_x_therm": np.array([1.0e34]),
                "S_x_rcs_abs": np.array([3.0e-12]),
                "S_x_bb_abs": np.array([2.0e-12]),
                "idx": np.array([0]),
            },
        },
        "expected_dfs_with_xrays": {
            "PMPS": pd.DataFrame(
                {
                    ("age", "[yr]"): [1e6, 2e6],
                    ("RA", "[deg]"): [180.0, 190.0],
                    ("DEC", "[deg]"): [45.0, 50.0],
                    ("l", "[deg]"): [120.0, 130.0],
                    ("b", "[deg]"): [30.0, 35.0],
                    ("DM", "[pc cm^-3]"): [10.0, 12.0],
                    ("d", "[kpc]"): [1.0, 1.5],
                    ("pm_RA", "[mas yr^-1]"): [3.0, 4.0],
                    ("pm_DEC", "[mas yr^-1]"): [2.0, 2.5],
                    ("v_ls", "[km s^-1]"): [100.0, 110.0],
                    ("B", "[G]"): [1e12, 1.1e12],
                    ("chi", "[rad]"): [0.1, 0.2],
                    ("P", "[s]"): [0.5, 0.6],
                    ("P_dot", "[s s^-1]"): [1e-15, 1.1e-15],
                    ("L_radio_bol", "[erg s^-1]"): [1e30, 1.1e30],
                    ("S_radio_obs_mean", "[Jy]"): [0.01, 0.02],
                    ("S_radio_obs_mean_1400", "[Jy]"): [0.005, 0.007],
                    ("w_int", "[s]"): [0.002, 0.003],
                    ("w_eff", "[s]"): [0.004, 0.005],
                    ("tau_sc", "[s]"): [0.001, 0.001],
                    ("spectral_index", ""): [-1.4, -1.5],
                }
            ),
            "HTRU_low_mid": pd.DataFrame(
                {
                    ("age", "[yr]"): [3e6, 4e6],
                    ("RA", "[deg]"): [200.0, 210.0],
                    ("DEC", "[deg]"): [55.0, 60.0],
                    ("l", "[deg]"): [140.0, 150.0],
                    ("b", "[deg]"): [40.0, 45.0],
                    ("DM", "[pc cm^-3]"): [14.0, 15.0],
                    ("d", "[kpc]"): [2.0, 2.5],
                    ("pm_RA", "[mas yr^-1]"): [5.0, 6.0],
                    ("pm_DEC", "[mas yr^-1]"): [3.0, 3.5],
                    ("v_ls", "[km s^-1]"): [120.0, 130.0],
                    ("B", "[G]"): [1.2e12, 1.3e12],
                    ("chi", "[rad]"): [0.3, 0.4],
                    ("P", "[s]"): [0.7, 0.8],
                    ("P_dot", "[s s^-1]"): [1.2e-15, 1.3e-15],
                    ("L_radio_bol", "[erg s^-1]"): [1.2e30, 1.3e30],
                    ("S_radio_obs_mean", "[Jy]"): [0.03, 0.04],
                    ("S_radio_obs_mean_1400", "[Jy]"): [0.008, 0.009],
                    ("w_int", "[s]"): [0.006, 0.007],
                    ("w_eff", "[s]"): [0.008, 0.009],
                    ("tau_sc", "[s]"): [0.001, 0.001],
                    ("spectral_index", ""): [-1.6, -1.7],
                    ("HTRU_low", ""): [True, False],
                    ("HTRU_mid", ""): [True, True],
                }
            ),
            "xray_flux_threshold": pd.DataFrame(
                {
                    ("age", "[yr]"): [1e4],
                    ("RA", "[deg]"): np.array([50.0]),
                    ("DEC", "[deg]"): np.array([-50.0]),
                    ("l", "[deg]"): np.array([-50.0]),
                    ("b", "[deg]"): np.array([-20.0]),
                    ("N_H", "[cm^-2]"): np.array([2.0e-21]),
                    ("d", "[kpc]"): np.array([2.0]),
                    ("pm_RA", "[mas yr^-1]"): np.array([-50.0]),
                    ("pm_DEC", "[mas yr^-1]"): np.array([-50.0]),
                    ("v_ls", "[km s^-1]"): np.array([-50.0]),
                    ("B_initial", "[G]"): np.array([1e12]),
                    ("B", "[G]"): np.array([1e12]),
                    ("chi", "[rad]"): np.array([1.0]),
                    ("P", "[s]"): np.array([0.01]),
                    ("P_dot", "[s s^-1]"): np.array([1.0e-11]),
                    ("L_x_therm", "[erg s^-1]"): np.array([1.0e34]),
                    ("S_x_rcs_abs", "[erg s^-1 cm^-2]"): np.array([3.0e-12]),
                    ("S_x_bb_abs", "[erg s^-1 cm^-2]"): np.array([2.0e-12]),
                },
            ),
            "xray_realistic": pd.DataFrame(
                {
                    ("age", "[yr]"): [1e4],
                    ("RA", "[deg]"): np.array([50.0]),
                    ("DEC", "[deg]"): np.array([-50.0]),
                    ("l", "[deg]"): np.array([-50.0]),
                    ("b", "[deg]"): np.array([-20.0]),
                    ("N_H", "[cm^-2]"): np.array([2.0e-21]),
                    ("d", "[kpc]"): np.array([2.0]),
                    ("pm_RA", "[mas yr^-1]"): np.array([-50.0]),
                    ("pm_DEC", "[mas yr^-1]"): np.array([-50.0]),
                    ("v_ls", "[km s^-1]"): np.array([-50.0]),
                    ("B_initial", "[G]"): np.array([1e12]),
                    ("B", "[G]"): np.array([1e12]),
                    ("chi", "[rad]"): np.array([1.0]),
                    ("P", "[s]"): np.array([0.01]),
                    ("P_dot", "[s s^-1]"): np.array([1.0e-11]),
                    ("L_x_therm", "[erg s^-1]"): np.array([1.0e34]),
                    ("S_x_rcs_abs", "[erg s^-1 cm^-2]"): np.array([3.0e-12]),
                    ("S_x_bb_abs", "[erg s^-1 cm^-2]"): np.array([2.0e-12]),
                }
            ),
        },
        "expected_dfs_without_xrays": {
            "PMPS": pd.DataFrame(
                {
                    ("age", "[yr]"): [1e6, 2e6],
                    ("RA", "[deg]"): [180.0, 190.0],
                    ("DEC", "[deg]"): [45.0, 50.0],
                    ("l", "[deg]"): [120.0, 130.0],
                    ("b", "[deg]"): [30.0, 35.0],
                    ("DM", "[pc cm^-3]"): [10.0, 12.0],
                    ("d", "[kpc]"): [1.0, 1.5],
                    ("pm_RA", "[mas yr^-1]"): [3.0, 4.0],
                    ("pm_DEC", "[mas yr^-1]"): [2.0, 2.5],
                    ("v_ls", "[km s^-1]"): [100.0, 110.0],
                    ("B", "[G]"): [1e12, 1.1e12],
                    ("chi", "[rad]"): [0.1, 0.2],
                    ("P", "[s]"): [0.5, 0.6],
                    ("P_dot", "[s s^-1]"): [1e-15, 1.1e-15],
                    ("L_radio_bol", "[erg s^-1]"): [1e30, 1.1e30],
                    ("S_radio_obs_mean", "[Jy]"): [0.01, 0.02],
                    ("S_radio_obs_mean_1400", "[Jy]"): [0.005, 0.007],
                    ("w_int", "[s]"): [0.002, 0.003],
                    ("w_eff", "[s]"): [0.004, 0.005],
                    ("tau_sc", "[s]"): [0.001, 0.001],
                    ("spectral_index", ""): [-1.4, -1.5],
                }
            ),
            "HTRU_low_mid": pd.DataFrame(
                {
                    ("age", "[yr]"): [3e6, 4e6],
                    ("RA", "[deg]"): [200.0, 210.0],
                    ("DEC", "[deg]"): [55.0, 60.0],
                    ("l", "[deg]"): [140.0, 150.0],
                    ("b", "[deg]"): [40.0, 45.0],
                    ("DM", "[pc cm^-3]"): [14.0, 15.0],
                    ("d", "[kpc]"): [2.0, 2.5],
                    ("pm_RA", "[mas yr^-1]"): [5.0, 6.0],
                    ("pm_DEC", "[mas yr^-1]"): [3.0, 3.5],
                    ("v_ls", "[km s^-1]"): [120.0, 130.0],
                    ("B", "[G]"): [1.2e12, 1.3e12],
                    ("chi", "[rad]"): [0.3, 0.4],
                    ("P", "[s]"): [0.7, 0.8],
                    ("P_dot", "[s s^-1]"): [1.2e-15, 1.3e-15],
                    ("L_radio_bol", "[erg s^-1]"): [1.2e30, 1.3e30],
                    ("S_radio_obs_mean", "[Jy]"): [0.03, 0.04],
                    ("S_radio_obs_mean_1400", "[Jy]"): [0.008, 0.009],
                    ("w_int", "[s]"): [0.006, 0.007],
                    ("w_eff", "[s]"): [0.008, 0.009],
                    ("tau_sc", "[s]"): [0.001, 0.001],
                    ("spectral_index", ""): [-1.6, -1.7],
                    ("HTRU_low", ""): [True, False],
                    ("HTRU_mid", ""): [True, True],
                }
            ),
        },
    }

    return data


def test_initialize_radio_surveys(test_case_1, monkeypatch):
    """
    Check that the radio surveys and the dictionaries containing the detected stars are properly initialized.
    """
    monkeypatch.setattr(sr, "SurveyRadio", MockSurveyRadio)
    (
        radio_surveys_out,
        detection_dicts_out,
    ) = sw.initialize_radio_surveys()

    # Verify the structure of returned dictionaries.
    # Assertions for radio_surveys.
    assert len(radio_surveys_out) == 3  # PMPS, HTRU_low, HTRU_mid
    assert isinstance(radio_surveys_out["PMPS"], MockSurveyRadio)
    assert isinstance(radio_surveys_out["HTRU_low"], MockSurveyRadio)
    assert isinstance(radio_surveys_out["HTRU_mid"], MockSurveyRadio)

    # Assertions for detection_dictionaries.
    assert len(detection_dicts_out) == 2  # PMPS, HTRU_low_mid
    for survey in test_case_1["detection_dicts_expected"].keys():
        assert survey in detection_dicts_out.keys()

    for key in test_case_1["detection_dicts_expected"]["PMPS"]:
        assert key in detection_dicts_out["PMPS"]
        assert detection_dicts_out["PMPS"][key] == []

    for key in test_case_1["detection_dicts_expected"]["HTRU_low_mid"]:
        assert key in detection_dicts_out["HTRU_low_mid"]
        assert detection_dicts_out["HTRU_low_mid"][key] == []


def test_initialize_xray_surveys(test_case_2, monkeypatch):
    """
    Check that the xray surveys and the dictionaries containing the detected stars are properly initialized.
    """
    monkeypatch.setattr(sx, "SurveyXray", MockSurveyXray)
    (
        xray_surveys_out,
        detection_dicts_out,
    ) = sw.initialize_xray_surveys()

    # Verify the structure of returned dictionaries.
    # Assertions for radio_surveys.
    assert len(xray_surveys_out) == 2  # xray_flux_threshold, xray_realistic
    assert isinstance(xray_surveys_out["xray_flux_threshold"], MockSurveyXray)
    assert isinstance(xray_surveys_out["xray_realistic"], MockSurveyXray)

    # Assertions for detection_dictionaries.
    assert len(detection_dicts_out) == 2  # xray_flux_threshold, xray_realistic
    for survey in test_case_2["detection_dicts_expected"].keys():
        assert survey in detection_dicts_out.keys()

    for key in test_case_2["detection_dicts_expected"]["xray_flux_threshold"]:
        assert key in detection_dicts_out["xray_flux_threshold"]
        assert detection_dicts_out["xray_flux_threshold"][key] == []

    for key in test_case_2["detection_dicts_expected"]["xray_realistic"]:
        assert key in detection_dicts_out["xray_realistic"]
        assert detection_dicts_out["xray_realistic"][key] == []


def test_initialize_all_surveys_without_xrays(
    test_case_1, test_case_2, test_case_3, monkeypatch
):
    """
    Check that all the surveys are properly initialized when cfg["simulation_xray"] = False.
    """
    cfg["simulation_xray"] = False

    def mock_initialize_radio_surveys(*args, **kwargs):
        return (
            test_case_1["radio_surveys_expected"],
            test_case_1["detection_dicts_expected"],
        )

    monkeypatch.setattr(
        sw, "initialize_radio_surveys", mock_initialize_radio_surveys
    )

    def mock_initialize_xray_surveys(*args, **kwargs):
        return (
            test_case_2["xray_surveys_expected"],
            test_case_2["detection_dicts_expected"],
        )

    monkeypatch.setattr(
        sw, "initialize_xray_surveys", mock_initialize_xray_surveys
    )

    result = sw.initialize_all_surveys()

    assert isinstance(result, sw.SurveyData)
    assert result == test_case_3["SurveyData_without_xray_expected"]


def test_initialize_all_surveys_with_xrays(
    test_case_1, test_case_2, test_case_3, monkeypatch
):
    """
    Check that all the surveys are properly initialized when cfg["simulation_xray"] = True.
    """
    cfg["simulation_xray"] = True

    def mock_initialize_radio_surveys(*args, **kwargs):
        return (
            test_case_1["radio_surveys_expected"],
            test_case_1["detection_dicts_expected"],
        )

    monkeypatch.setattr(
        sw, "initialize_radio_surveys", mock_initialize_radio_surveys
    )

    def mock_initialize_xray_surveys(*args, **kwargs):
        return (
            test_case_2["xray_surveys_expected"],
            test_case_2["detection_dicts_expected"],
        )

    monkeypatch.setattr(
        sw, "initialize_xray_surveys", mock_initialize_xray_surveys
    )

    result = sw.initialize_all_surveys()

    assert isinstance(result, sw.SurveyData)
    assert result == test_case_3["SurveyData_with_xray_expected"]


def test_apply_surveys_coverage(test_case_4):
    """
    Check that the survey coverage filter is properly applied.
    """
    xray_surveys = None
    (
        dictionary_coverage_database,
        updated_idx_remove,
    ) = sw.apply_surveys_coverage(
        test_case_4["radio_surveys"],
        xray_surveys,
        test_case_4["dyn_database_dict"],
        test_case_4["idx_remove"],
        test_case_4["dist_cutoff"],
    )

    assert len(dictionary_coverage_database["age"]) <= len(
        test_case_4["dyn_database_dict"]["age"]
    )
    assert all(
        dictionary_coverage_database["dist"] < test_case_4["dist_cutoff"]
    )
    assert len(updated_idx_remove) == len(
        test_case_4["dyn_database_dict"]["idx"]
    ) - len(dictionary_coverage_database["idx"])
    # Check that dictionary keys match expectations.
    for key in test_case_4["expected_keys_without_xray"]:
        assert key in dictionary_coverage_database

    xray_surveys = test_case_4["xray_surveys"]
    test_case_4["idx_remove"] = []
    (
        dictionary_coverage_database,
        updated_idx_remove,
    ) = sw.apply_surveys_coverage(
        test_case_4["radio_surveys"],
        xray_surveys,
        test_case_4["dyn_database_dict"],
        test_case_4["idx_remove"],
        test_case_4["dist_cutoff"],
    )

    assert len(dictionary_coverage_database["age"]) <= len(
        test_case_4["dyn_database_dict"]["age"]
    )
    assert all(
        dictionary_coverage_database["dist"] < test_case_4["dist_cutoff"]
    )
    assert len(updated_idx_remove) == len(
        test_case_4["dyn_database_dict"]["idx"]
    ) - len(dictionary_coverage_database["idx"])
    # Check that dictionary keys match expectations.
    for key in test_case_4["expected_keys_with_xray"]:
        assert key in dictionary_coverage_database


def test_radio_detection(test_case_5):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in radio is properly returned.
    """
    output_dict = sw.radio_detection(
        test_case_5["radio_surveys"],
        test_case_5["dictionary_intercepted_radio"],
    )
    # Verify that the keys are correct.
    assert set(output_dict.keys()) == set(
        test_case_5["expected_update_dictionary_detected"].keys()
    )
    for key in test_case_5["expected_update_dictionary_detected"]:
        assert set(output_dict[key].keys()) == set(
            test_case_5["expected_update_dictionary_detected"][key].keys()
        )

    # Validate overlapping logic for HTRU_low and HTRU_mid.
    assert len(output_dict["HTRU_low_mid"]["age"]) == 1
    assert (
        output_dict["HTRU_low_mid"]["HTRU_low"]
        == test_case_5["expected_update_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_low"
        ]
    )
    assert (
        output_dict["HTRU_low_mid"]["HTRU_mid"]
        == test_case_5["expected_update_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_mid"
        ]
    )


def test_xray_detection(test_case_6):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in X-ray is properly returned.
    """
    output_dict = sw.xray_detection(
        test_case_6["xray_surveys"],
        test_case_6["dictionary_xray_pop"],
    )
    # Verify that the keys are correct.
    assert set(output_dict.keys()) == set(
        test_case_6["update_dictionary_detected_x_expected"].keys()
    )
    for key in test_case_6["update_dictionary_detected_x_expected"]:
        assert set(output_dict[key].keys()) == set(
            test_case_6["update_dictionary_detected_x_expected"][key].keys()
        )


def test_update_filtered_dictionary(test_case_7):
    """
    Check that the method to update the dictionary of detected neutron stars work properly.
    """

    result = sw.update_filtered_dictionary(
        test_case_7["dict_to_update"],
        test_case_7["detected_mask"],
        **test_case_7["additional_properties"]
    )

    # Assert the result matches the expected output.
    for key in test_case_7["expected_output"]:
        assert result[key] == test_case_7["expected_output"][key]


def test_update_survey_data(test_case_8):
    """
    Check that the a SurveyData object is properly updated.
    """

    # Test if updating the radio survey data works properly.
    survey_data = test_case_8["SurveyData"]
    logger = MagicMock()

    idx_remove = []

    sw.update_survey_data(
        survey_data,
        test_case_8["update_dictionary_detected_radio"],
        "radio",
        1000,
        idx_remove,
        logger,
    )

    # Check updated detection count.
    assert survey_data.n_detected_sim["PMPS"] == 1
    assert survey_data.n_detected_sim["HTRU_low_mid"] == 1

    # Check dictionary updated
    assert (
        survey_data.dictionary_detected_radio
        == test_case_8["dictionary_detected_radio_expected"]
    )

    # Check idx_remove updated.
    assert idx_remove == test_case_8["idx_remove_expected"]

    logger.info.assert_has_calls(
        [
            call("Total number of neutron stars detected by PMPS: 1"),
            call("Total number of neutron stars detected by HTRU_low_mid: 1"),
        ],
        any_order=True,
    )

    # Still under detection threshold.
    assert survey_data.stop_flags["PMPS"] is False
    assert survey_data.n_detected_sim_at_match["PMPS"] == 0
    assert survey_data.n_created_at_match["PMPS"] == 0

    # Test if updating the X-ray survey data works properly.
    survey_data = test_case_8["SurveyData"]
    logger = MagicMock()

    idx_remove = []

    sw.update_survey_data(
        survey_data,
        test_case_8["update_dictionary_detected_xray"],
        "X-ray",
        1000,
        idx_remove,
        logger,
    )

    # Check updated detection count.
    assert survey_data.n_detected_sim["xray_flux_threshold"] == 1
    assert survey_data.n_detected_sim["xray_realistic"] == 1

    # Check dictionary updated
    assert (
        survey_data.dictionary_detected_xray
        == test_case_8["dictionary_detected_xray_expected"]
    )

    # Check idx_remove updated.
    assert idx_remove == test_case_8["idx_remove_expected"]

    logger.info.assert_has_calls(
        [
            call(
                "Total number of neutron stars detected by xray_flux_threshold: 1"
            ),
            call(
                "Total number of neutron stars detected by xray_realistic: 1"
            ),
        ],
        any_order=True,
    )

    # Still under detection threshold.
    assert survey_data.stop_flags["xray_realistic"] is False
    assert survey_data.n_detected_sim_at_match["xray_realistic"] == 0
    assert survey_data.n_created_at_match["xray_realistic"] == 0


def test_build_dataframe(test_case_9):
    """
    Check that the function to create a dataframe is properly working.
    """
    result_df = sw.build_dataframe(
        test_case_9["data_dict"],
        test_case_9["parameters"],
        test_case_9["units"],
    )

    # Assert the result matches the expected output.
    pd.testing.assert_frame_equal(result_df, test_case_9["expected_df"])


def test_create_output_dataframe(test_case_10):
    """
    Check that the dataframe with the output of the survey detection are correct.
    """
    output_dfs_without_xray = sw.create_output_dataframe(
        test_case_10["dictionary_detected_radio"],
        None,
    )

    # Assert the result matches the expected output.
    for survey_name, expected_df in output_dfs_without_xray.items():
        pd.testing.assert_frame_equal(
            output_dfs_without_xray[survey_name],
            test_case_10["expected_dfs_without_xrays"][survey_name],
        )

    output_dfs_with_xray = sw.create_output_dataframe(
        test_case_10["dictionary_detected_radio"],
        test_case_10["dictionary_detected_xray"],
    )

    # Assert the result matches the expected output.
    for survey_name, expected_df in output_dfs_with_xray.items():
        pd.testing.assert_frame_equal(
            output_dfs_with_xray[survey_name],
            test_case_10["expected_dfs_with_xrays"][survey_name],
        )
