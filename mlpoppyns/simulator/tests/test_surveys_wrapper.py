"""
    Tests for the surveys_wrapper module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

import mlpoppyns.simulator.multiband_surveys.survey_radio as sr
import mlpoppyns.simulator.multiband_surveys.surveys_wrapper as sw
from mlpoppyns.simulator.config_simulator import cfg

# Set some radio surveys in the configuration file for testing purposes.
cfg["surveys_radio"]: dict = {
    "PMPS": {
        "path": "mlpoppyns/simulator/multiband_surveys/Parkes_parameters.json",
        "detected_real": 1045,
    },
    "HTRU_low_mid": {
        "path_low": "mlpoppyns/simulator/multiband_surveys/htru_low_parameters.json",
        "path_mid": "mlpoppyns/simulator/multiband_surveys/htru_mid_parameters.json",
        "detected_real": 1037,
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
        intercepted_radio,
        coverage,
        l_gal,
        b_gal,
        DEC,
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

    def detected_radio_population_full(
        self,
        w_int_s,
        DM,
        P,
        intercepted_radio,
        coverage,
        l_gal,
        b_gal,
        DEC,
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
def test_case_3():
    data = {
        "SurveyData_expected": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "mlpoppyns/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "mlpoppyns/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "mlpoppyns/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            n_detected_sim={survey: 0 for survey in ("PMPS", "HTRU_low_mid")},
            n_detected_complete_sim={
                survey: 0 for survey in ("PMPS", "HTRU_low_mid")
            },
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
        "coverage_dict": {
            "coverage_radio": np.array([True, False]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
        },
        "idx_remove": [],
        "dist_cutoff": 5.0,
        "expected_keys": {
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
        "dist_cutoff": 5.0,
        "expected_keys": {
            "coverage_radio",
            "coverage_radio_PMPS",
            "coverage_radio_HTRU_low",
            "coverage_radio_HTRU_mid",
        },
    }

    return data


@pytest.fixture()
def test_case_6():
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
def test_case_7():
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
            "intercepted_radio": np.array([True, True]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
        },
        "expected_dictionary_detected": {
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
                "idx": np.array([0]),
                "coverage_radio_PMPS": np.array([True]),
                "coverage_radio_HTRU_low": np.array([True]),
                "coverage_radio_HTRU_mid": np.array([True]),
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
                "idx": np.array([0]),
                "HTRU_low": np.array([True]),
                "HTRU_mid": np.array([True]),
                "coverage_radio_PMPS": np.array([True]),
                "coverage_radio_HTRU_low": np.array([True]),
                "coverage_radio_HTRU_mid": np.array([True]),
            },
        },
    }

    return data


@pytest.fixture()
def test_case_9():
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
def test_case_10():
    data = {
        "SurveyData": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "mlpoppyns/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "mlpoppyns/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "mlpoppyns/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            n_detected_sim={survey: 0 for survey in ("PMPS", "HTRU_low_mid")},
            n_detected_complete_sim={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                )
            },
            percentage_detected={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                )
            },
            n_created_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                )
            },
            n_detected_sim_at_match={
                survey: 0
                for survey in (
                    "PMPS",
                    "HTRU_low_mid",
                )
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
                    "spectral_index": [],
                    "tau_sc": [],
                    "idx": [],
                    "HTRU_low": [],
                    "HTRU_mid": [],
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
    }

    return data


@pytest.fixture()
def test_case_11():
    data = {
        "dictionary_detected_radio": {
            "PMPS": {
                "idx": [0, 1],
                "age": [1e6, 2e6],
                "ra": [180.0, 190.0],
                "dec": [45.0, 50.0],
                "l": [120.0, 130.0],
                "b": [30.0, 35.0],
                "DM": [10.0, 12.0],
                "dist": [1.0, 1.5],
                "pm_ra": [3.0, 4.0],
                "pm_dec": [2.0, 2.5],
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
                "idx": [0, 1],
                "age": [3e6, 4e6],
                "ra": [200.0, 210.0],
                "dec": [55.0, 60.0],
                "l": [140.0, 150.0],
                "b": [40.0, 45.0],
                "DM": [14.0, 15.0],
                "dist": [2.0, 2.5],
                "pm_ra": [5.0, 6.0],
                "pm_dec": [3.0, 3.5],
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
                "HTRU_low": [True, False],
                "HTRU_mid": [True, True],
            },
        },
        "expected_dfs": {
            "PMPS": pd.DataFrame(
                {
                    ("idx", ""): [0, 1],
                    ("age", "[yr]"): [1e6, 2e6],
                    ("ra", "[deg]"): [180.0, 190.0],
                    ("dec", "[deg]"): [45.0, 50.0],
                    ("l", "[deg]"): [120.0, 130.0],
                    ("b", "[deg]"): [30.0, 35.0],
                    ("DM", "[pc cm^-3]"): [10.0, 12.0],
                    ("dist", "[kpc]"): [1.0, 1.5],
                    ("pm_ra", "[mas yr^-1]"): [3.0, 4.0],
                    ("pm_dec", "[mas yr^-1]"): [2.0, 2.5],
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
                    ("idx", ""): [0, 1],
                    ("age", "[yr]"): [3e6, 4e6],
                    ("ra", "[deg]"): [200.0, 210.0],
                    ("dec", "[deg]"): [55.0, 60.0],
                    ("l", "[deg]"): [140.0, 150.0],
                    ("b", "[deg]"): [40.0, 45.0],
                    ("DM", "[pc cm^-3]"): [14.0, 15.0],
                    ("dist", "[kpc]"): [2.0, 2.5],
                    ("pm_ra", "[mas yr^-1]"): [5.0, 6.0],
                    ("pm_dec", "[mas yr^-1]"): [3.0, 3.5],
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


@pytest.fixture()
def test_case_12():
    data = {
        "SurveyData": sw.SurveyData(
            surveys_cfg={
                "PMPS": {
                    "path": "mlpoppyns/simulator/multiband_surveys/Parkes_parameters.json",
                    "detected_real": 1045,
                },
                "HTRU_low_mid": {
                    "path_low": "mlpoppyns/simulator/multiband_surveys/htru_low_parameters.json",
                    "path_mid": "mlpoppyns/simulator/multiband_surveys/htru_mid_parameters.json",
                    "detected_real": 1037,
                },
            },
            surveys_radio={
                "PMPS": MockSurveyRadio("PMPS"),
                "HTRU_low": MockSurveyRadio("HTRU_low"),
                "HTRU_mid": MockSurveyRadio("HTRU_mid"),
            },
            n_detected_sim={survey: 0 for survey in ("PMPS", "HTRU_low_mid")},
            n_detected_complete_sim={
                survey: 0 for survey in ("PMPS", "HTRU_low_mid")
            },
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
        ),
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


def test_initialize_all_surveys(test_case_1, test_case_3, monkeypatch):
    """
    Check that all the surveys are properly initialized.
    """

    def mock_initialize_radio_surveys(*args, **kwargs):
        return (
            test_case_1["radio_surveys_expected"],
            test_case_1["detection_dicts_expected"],
        )

    monkeypatch.setattr(
        sw, "initialize_radio_surveys", mock_initialize_radio_surveys
    )

    result = sw.initialize_all_surveys()

    assert isinstance(result, sw.SurveyData)
    assert result == test_case_3["SurveyData_expected"]


def test_apply_surveys_coverage_filter(test_case_4):
    """
    Check that the survey coverage filter is properly applied.
    """
    (
        dictionary_coverage_database,
        updated_idx_remove,
    ) = sw.apply_surveys_coverage_filter(
        test_case_4["radio_surveys"],
        test_case_4["coverage_dict"],
        test_case_4["dyn_database_dict"],
        test_case_4["idx_remove"],
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
    for key in test_case_4["expected_keys"]:
        assert key in dictionary_coverage_database


def test_compute_surveys_coverage(test_case_5):
    """
    Check that the survey coverage dictionary is properly computed.
    """

    dictionary_coverage = sw.compute_surveys_coverage(
        test_case_5["radio_surveys"],
        test_case_5["dyn_database_dict"],
        test_case_5["dist_cutoff"],
    )

    assert len(dictionary_coverage["coverage_radio"]) == len(
        test_case_5["dyn_database_dict"]["age"]
    )

    # Check that dictionary keys match expectations.
    for key in test_case_5["expected_keys"]:
        assert key in dictionary_coverage


def test_radio_detection(test_case_6):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in radio is properly returned.
    """

    logger = MagicMock()

    output_dict = sw.radio_detection(
        test_case_6["radio_surveys"],
        test_case_6["dictionary_intercepted_radio"],
        np.ones(
            len(test_case_6["dictionary_intercepted_radio"]["w_int"]),
            dtype=bool,
        ),
        logger,
    )
    # Verify that the keys are correct.
    assert set(output_dict.keys()) == set(
        test_case_6["expected_update_dictionary_detected"].keys()
    )
    for key in test_case_6["expected_update_dictionary_detected"]:
        assert set(output_dict[key].keys()) == set(
            test_case_6["expected_update_dictionary_detected"][key].keys()
        )

    # Validate overlapping logic for HTRU_low and HTRU_mid.
    assert len(output_dict["HTRU_low_mid"]["age"]) == 1
    assert (
        output_dict["HTRU_low_mid"]["HTRU_low"]
        == test_case_6["expected_update_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_low"
        ]
    )
    assert (
        output_dict["HTRU_low_mid"]["HTRU_mid"]
        == test_case_6["expected_update_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_mid"
        ]
    )


def test_radio_detection_full(test_case_7):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in radio is properly returned
    for the full simulation.
    """
    logger = MagicMock()

    output_dict = sw.radio_detection(
        test_case_7["radio_surveys"],
        test_case_7["dictionary_intercepted_radio"],
        test_case_7["dictionary_intercepted_radio"]["intercepted_radio"],
        logger,
        full_population=True,
    )
    # Verify that the keys are correct.
    assert set(output_dict.keys()) == set(
        test_case_7["expected_dictionary_detected"].keys()
    )
    for key in test_case_7["expected_dictionary_detected"]:
        assert set(output_dict[key].keys()) == set(
            test_case_7["expected_dictionary_detected"][key].keys()
        )

    # Validate overlapping logic for HTRU_low and HTRU_mid.
    assert len(output_dict["HTRU_low_mid"]["age"]) == 1
    assert (
        output_dict["HTRU_low_mid"]["HTRU_low"]
        == test_case_7["expected_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_low"
        ]
    )
    assert (
        output_dict["HTRU_low_mid"]["HTRU_mid"]
        == test_case_7["expected_dictionary_detected"]["HTRU_low_mid"][
            "HTRU_mid"
        ]
    )


def test_update_filtered_dictionary(test_case_9):
    """
    Check that the method to update the dictionary of detected neutron stars works properly.
    """

    result = sw.update_filtered_dictionary(
        test_case_9["dict_to_update"],
        test_case_9["detected_mask"],
        **test_case_9["additional_properties"]
    )

    # Assert the result matches the expected output.
    for key in test_case_9["expected_output"]:
        assert result[key] == test_case_9["expected_output"][key]


def test_update_survey_data(test_case_10):
    """
    Check that the SurveyData object is properly updated.
    """

    # Test if updating the radio survey data works properly.
    survey_data = test_case_10["SurveyData"]
    logger = MagicMock()

    idx_remove = []

    sw.update_survey_data(
        survey_data,
        test_case_10["update_dictionary_detected_radio"],
        "radio",
        1000,
        idx_remove,
        logger,
    )

    # Check updated detection count.
    assert survey_data.n_detected_sim["PMPS"] == 1
    assert survey_data.n_detected_sim["HTRU_low_mid"] == 1

    # Check dictionary updated.
    assert (
        survey_data.dictionary_detected_radio
        == test_case_10["dictionary_detected_radio_expected"]
    )

    # Check idx_remove updated.
    assert idx_remove == test_case_10["idx_remove_expected"]

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


def test_create_output_dataframe(test_case_11):
    """
    Check that the dataframe with the output of the survey detection is correct.
    """
    output_dfs = sw.create_output_dataframe_surveys(
        test_case_11["dictionary_detected_radio"]
    )

    # Assert the result matches the expected output.
    for survey_name, expected_df in output_dfs.items():
        pd.testing.assert_frame_equal(
            output_dfs[survey_name],
            test_case_11["expected_dfs"][survey_name],
        )


def test_adjust_n_batchsize(test_case_12):
    """
    Check that the batch size for the simulation is correctly adjusted.
    """
    survey_data = test_case_12["SurveyData"]

    survey_data.n_detected_complete_sim["PMPS"] = 1
    survey_data.n_detected_complete_sim["HTRU_low-mid"] = 1

    n_batchsize = sw.adjust_n_batchsize(survey_data)

    # Assertions.
    assert n_batchsize == 100000
    assert survey_data.batchsize_adjust_flags[0.9] is False
    assert survey_data.batchsize_adjust_flags[0.95] is False

    survey_data.n_detected_complete_sim["PMPS"] = 950
    survey_data.n_detected_complete_sim["HTRU_low_mid"] = 950

    n_batchsize = sw.adjust_n_batchsize(survey_data)

    # Assertions.
    assert n_batchsize == 10000
    assert survey_data.batchsize_adjust_flags[0.9] is True
    assert survey_data.batchsize_adjust_flags[0.95] is False

    survey_data.n_detected_complete_sim["PMPS"] = 1030
    survey_data.n_detected_complete_sim["HTRU_low_mid"] = 1030

    n_batchsize = sw.adjust_n_batchsize(survey_data)

    # Assertions.
    assert n_batchsize == 5000
    assert survey_data.batchsize_adjust_flags[0.9] is True
    assert survey_data.batchsize_adjust_flags[0.95] is True
