"""
    Tests for the simulate_population_magrot_det module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import json
import pathlib
import pickle
from unittest import mock
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import RectBivariateSpline

import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.multiband_emission.emission_xray as ex
import pypopsyn.simulator.multiband_surveys.survey_radio as sr
import pypopsyn.simulator.simulate_population_magrot_det as sim
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.samplers.memory_efficient_sampling as mes
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


class MockSurveyRadio:
    """Mock class to simulate a radio survey."""

    def __init__(self, name):
        self.name = name

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
        """Mock implementation of sky_coverage."""
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
        "dyn_path": pathlib.Path("/mock/path/to/"),
        "dyn_data_path": pathlib.Path("/mock/path/to/final_pop_dyn.csv"),
        "dyn_config_path": pathlib.Path("/mock/path/to/configuration.json"),
        "n_batchsize": 10,
        "NS_number": 100,
        "idx_remove": [1, 3, 5],
        "mock_df_dyn": pd.DataFrame(
            {
                ("age", "[yr]"): [1e6, 2e6],
                ("r", "[kpc]"): [8.0, 8.5],
                ("phi", "[rad]"): [0.1, 0.2],
                ("z", "[kpc]"): [0.05, 0.1],
                ("v_r", "[km/s]"): [100, 200],
                ("v_phi", "[km/s]"): [220, 230],
                ("v_z", "[km/s]"): [10, 20],
            }
        ),
        "mock_polar_to_cartesian": (
            np.array([7.96, 8.46]),
            np.array([0.8, 1.7]),
        ),
        "mock_speed_cylindrical_to_cartesian": (
            np.array([110, 210]),
            np.array([210, 220]),
            np.array([15, 25]),
        ),
        "mock_galactocentric_to_icrs": (
            np.array([180.0, 181.0]),
            np.array([-30.0, -31.0]),
            np.array([8.0, 8.5]),
            np.array([5.0, 5.1]),
            np.array([-2.0, -2.1]),
            np.array([300, 310]),
        ),
        "mock_galactocentric_to_galactic": (
            np.array([50.0, 51.0]),
            np.array([10.0, 11.0]),
            np.array([8.0, 8.5]),
            np.array([1.0, 1.1]),
            np.array([-0.5, -0.6]),
            np.array([250, 260]),
        ),
        "config_dyn": {
            "t_age_max": 1.0e8,
            "NS_number": 100,
            "kick_model": "kick_model",
            "sigma_k": 100.0,
            "vk_c": 100.0,
            "h_c": 100.0,
        },
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
        },
        "cfg_expected": {
            "t_age_max": 1.0e8,
            "NS_number": 100,
            "kick_model": "kick_model",
            "sigma_k": 100.0,
            "vk_c": 100.0,
            "h_c": 100.0,
            "dyn_database_path": "/mock/path/to",
        },
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "dict_coverage_database": {
            "age": np.array([1e6, 2e6]),
        },
        "expected_keys": {
            "age",
            "B_initial",
            "P_initial",
            "chi_initial",
        },
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "dict_pop_initial_magrot": {
            "age": np.array([1e6]),
            "B_initial": np.array([1.0e12]),
            "P_initial": np.array([0.1]),
            "chi_initial": np.array([0.5]),
        },
        "B_final": np.array([1.0e12]),
        "P_final": np.array([0.1]),
        "chi_final": np.array([0.5]),
        "magrot_evol_dict": {
            "0": {
                "t": [],
                "B(t)": [],
                "P(t)": [],
                "chi(t)": [],
            }
        },
        "expected_keys": {
            "B_initial",
            "B",
            "P",
            "P_dot",
            "chi",
        },
    }

    return data


@pytest.fixture()
def test_case_6():
    data = {
        "dict_final_pop": {
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
            "B": np.array([1e12, 1e14]),
            "chi": np.array([1.0, 2.0]),
            "idx": np.array([0, 1]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
            "coverage_radio": np.array([True, False]),
        },
        "w_int_s": np.array([0.001]),
        "L_radio_bol": np.array([1.0e26]),
        "S_radio_bol": np.array([1.0e-6]),
        "spectral_index": np.array([-1.8]),
        "DM": np.array([100]),
        "tau_sc": np.array([0.001]),
        "intercepted_radio": np.array([True]),
        "expected_keys": [
            "age",
            "l",
            "b",
            "ra",
            "dec",
            "dist",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "B",
            "chi",
            "P",
            "P_dot",
            "w_int",
            "DM",
            "idx",
            "L_radio_bol",
            "S_radio_bol",
            "spectral_index",
            "tau_sc",
            "coverage_radio_PMPS",
            "coverage_radio_HTRU_low",
            "coverage_radio_HTRU_mid",
            "coverage_radio",
        ],
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
                "spectral_index": [-1.6, -1.7],
                ("HTRU_low", ""): [True, False],
                ("HTRU_mid", ""): [True, True],
            },
        },
        "dictionary_flux_threshold_x": {
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
            "B": np.array([1e12]),
            "chi": np.array([1.0]),
            "P": np.array([0.01]),
            "P_dot": np.array([1.0e-11]),
            "L_x_therm": np.array([1.0e34]),
            "S_x_rcs_abs": np.array([3.0e-12]),
            "S_x_bb_abs": np.array([2.0e-12]),
            "idx": np.array([0]),
        },
        "dictionary_detected_x": {
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
            "B": np.array([1e12]),
            "chi": np.array([1.0]),
            "P": np.array([0.01]),
            "P_dot": np.array([1.0e-11]),
            "L_x_therm": np.array([1.0e34]),
            "S_x_rcs_abs": np.array([3.0e-12]),
            "S_x_bb_abs": np.array([2.0e-12]),
            "idx": np.array([0]),
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
                    ("spectral_index", ""): [-1.6, -1.7],
                    ("HTRU_low", ""): [True, False],
                    ("HTRU_mid", ""): [True, True],
                }
            ),
            "X-ray_flux_threshold": pd.DataFrame(
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
                    ("B", "[G]"): np.array([1e12]),
                    ("chi", "[rad]"): np.array([1.0]),
                    ("P", "[s]"): np.array([0.01]),
                    ("P_dot", "[s s^-1]"): np.array([1.0e-11]),
                    ("L_x_therm", "[erg s^-1]"): np.array([1.0e34]),
                    ("S_x_rcs_abs", "[erg s^-1 cm^-2]"): np.array([3.0e-12]),
                    ("S_x_bb_abs", "[erg s^-1 cm^-2]"): np.array([2.0e-12]),
                },
            ),
            "X-ray_detected": pd.DataFrame(
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
                    ("spectral_index", ""): [-1.6, -1.7],
                    ("HTRU_low", ""): [True, False],
                    ("HTRU_mid", ""): [True, True],
                }
            ),
        },
    }

    return data


@pytest.fixture()
def test_case_11():
    data = {
        "mock_L_x_interpolator": RectBivariateSpline(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        ),
        "dictionary_detected_x_expected": {
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
            "S_x_rcs_abs": [],
            "S_x_bb_abs": [],
            "idx": [],
        },
    }

    return data


@pytest.fixture()
def test_case_12():
    data = {
        "dict_final_pop": {
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
            "idx": np.array([0, 1]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
            "coverage_radio": np.array([True, False]),
            "coverage_x": np.array([True, True]),
        },
        "L_x_threshold": 1e29,
        "S_x_abs_threshold": 1e-15,
        "dummy_L_x_interpolator": RectBivariateSpline(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        ),
        "mock_xray_bright_mask": np.array([True, True]),
        "mock_L_x_therm": np.array([1e33, 1e34]),
        "mock_S_x_rcs_abs": np.array([3.0e-12, 4.0e-16]),
        "mock_S_x_bb_abs": np.array([2.0e-12, 3.0e-16]),
        "mock_N_H": np.array([2.0e-21, 3.0e-21]),
        "detected_x_expected": np.array([True, False]),
        "update_dictionary_detected_x_expected": {
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
            "coverage_x": np.array([True]),
        },
    }

    return data


def test_initialize_x_surveys(test_case_11, monkeypatch):
    """
    Check that the x survey and the dictionaries containing the detected stars are properly initialized.
    """

    # Mocking pathlib and pickle.
    with patch(
        "builtins.open",
        mock_open(
            read_data=pickle.dumps(test_case_11["mock_L_x_interpolator"])
        ),
    ):
        with patch(
            "pickle.load", return_value=test_case_11["mock_L_x_interpolator"]
        ):
            with patch(
                "pathlib.Path.joinpath",
                return_value=pathlib.Path("mock_path.pkl"),
            ):

                (
                    dictionary_flux_threshold_x,
                    dictionary_detected_x,
                    L_x_interpolator,
                ) = sim.initialize_x_surveys()

                # Check the returned types.
                assert isinstance(
                    dictionary_flux_threshold_x, dict
                ), "Expected a dictionary"
                assert isinstance(
                    dictionary_detected_x, dict
                ), "Expected a dictionary"
                assert isinstance(
                    L_x_interpolator, RectBivariateSpline
                ), "Expected a RectBivariateSpline interpolator"

                # Check that dictionary keys match expectations.
                for key in test_case_11["dictionary_detected_x_expected"]:
                    assert key in dictionary_flux_threshold_x
                    assert dictionary_flux_threshold_x[key] == []
                for key in test_case_11["dictionary_detected_x_expected"]:
                    assert key in dictionary_detected_x
                    assert dictionary_detected_x[key] == []


def test_xray_detection(test_case_12, monkeypatch):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in X-rays is properly returned.
    """

    def mock_calculate_xray_emission(*args, **kwargs):
        return (
            test_case_12["mock_xray_bright_mask"],
            test_case_12["mock_L_x_therm"],
            test_case_12["mock_S_x_bb_abs"],
            test_case_12["mock_S_x_rcs_abs"],
            test_case_12["mock_N_H"],
        )

    monkeypatch.setattr(
        ex, "calculate_xray_emission", mock_calculate_xray_emission
    )

    output_dict_flux_threshold, output_dict_detected = sim.xray_detection(
        test_case_12["dict_final_pop"],
        test_case_12["dummy_L_x_interpolator"],
        test_case_12["L_x_threshold"],
        test_case_12["S_x_abs_threshold"],
    )
    # Verify that the keys are correct.
    assert set(output_dict_flux_threshold.keys()) == set(
        test_case_12["update_dictionary_detected_x_expected"].keys()
    )
    assert set(output_dict_detected.keys()) == set(
        test_case_12["update_dictionary_detected_x_expected"].keys()
    )
