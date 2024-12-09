"""
    Tests for the simulate_population_magrot_det module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import pathlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
import pypopsyn.simulator.multiband_emission.emission_radio as er
import pypopsyn.simulator.simulate_population_magrot_det_v2 as sim
import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.samplers.memory_efficient_sampling as mes
from pypopsyn.simulator.config_simulator import cfg


@pytest.fixture()
def test_case_1():
    data = {
        "dyn_path_pop": pathlib.Path("/mock/path/to/dynamic_pop.csv"),
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
        "expected_keys": {
            "age",
            "ra",
            "dec",
            "l",
            "b",
            "dist_heliocentric",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "idx",
        },
    }

    return data


class MockRadioSurvey:
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
def test_case_2():
    data = {
        "radio_surveys": {
            "PMPS": MockRadioSurvey("PMPS"),
            "SMPS": MockRadioSurvey("SMPS"),
            "HTRU_low": MockRadioSurvey("HTRU_low"),
            "HTRU_mid": MockRadioSurvey("HTRU_mid"),
            "HTRU_high": MockRadioSurvey("HTRU_high"),
        },
        "dyn_database_dict": {
            "age": np.array([1e6, 2e6]),
            "ra": np.array([50.0, 250.0]),
            "dec": np.array([-50.0, 50.0]),
            "l": np.array([-50.0, 50.0]),
            "b": np.array([-20.0, 10.0]),
            "dist_heliocentric": np.array([2.0, 10.0]),
            "pm_ra": np.array([-50.0, 50.0]),
            "pm_dec": np.array([-50.0, 50.0]),
            "v_ls": np.array([-50.0, 50.0]),
            "idx": np.array([0, 1]),
        },
        "idx_remove": [],
        "dist_cutoff": 5.0,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "dict_coverage_database": {
            "age": np.array([1e6, 2e6]),
        },
        "expected_keys": {
            "B_initial",
            "P_initial",
            "chi_initial",
        },
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "dict_coverage_database": {
            "age": np.array([1e6]),
        },
        "dict_pop_initial_magrot": {
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
            "B_final",
            "P_final",
            "chi_final",
        },
    }

    return data


@pytest.fixture()
def test_case_5():
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
            "P_final": np.array([0.01, 0.5]),
            "B_final": np.array([1e12, 1e14]),
            "chi_final": np.array([1.0, 2.0]),
            "idx": np.array([0, 1]),
            "coverage_PMPS": np.array([True, False]),
            "coverage_SMPS": np.array([True, False]),
            "coverage_HTRU_low": np.array([True, False]),
            "coverage_HTRU_mid": np.array([True, False]),
            "coverage_HTRU_high": np.array([True, False]),
            "coverage_radio": np.array([True, False]),
        },
        "mock_dictionary_intercepted_radio": {
            "age": np.array([1e6]),
            "l": np.array([-50.0]),
            "b": np.array([-20.0]),
            "B": np.array([1e12]),
            "chi": np.array([1.0]),
            "P": np.array([0.01]),
            "P_dot": np.array([1.0e-14]),
            "w_int_s": np.array([0.001]),
            "L_radio_bol": np.array([1.0e26]),
            "S_radio_bol": np.array([1.0e-6]),
            "spectral_index": np.array([-1.8]),
            "DM": np.array([100]),
            "tau_sc": np.array([0.001]),
            "idx": np.array([0]),
            "intercepted_radio": np.array([True]),
        },
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
            "w_int_s",
            "DM",
            "idx_radio",
            "L_radio_bol",
            "S_radio_bol",
            "spectral_index",
            "tau_sc",
            "coverage_PMPS",
            "coverage_SMPS",
            "coverage_HTRU_low",
            "coverage_HTRU_mid",
            "coverage_HTRU_high",
        ],
    }

    return data


@pytest.fixture()
def test_case_6():
    data = {
        "radio_surveys": {
            "PMPS": MockRadioSurvey("PMPS"),
            "SMPS": MockRadioSurvey("SMPS"),
            "HTRU_low": MockRadioSurvey("HTRU_low"),
            "HTRU_mid": MockRadioSurvey("HTRU_mid"),
            "HTRU_high": MockRadioSurvey("HTRU_high"),
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
            "w_int_s": np.array([0.05, 0.1]),
            "DM": np.array([100.0, 200.0]),
            "idx_radio": np.array([0, 1]),
            "L_radio_bol": np.array([1e30, 1e31]),
            "S_radio_bol": np.array([0.01, 0.02]),
            "spectral_index": np.array([-1.8, -1.8]),
            "tau_sc": np.array([0.001, 0.002]),
            "coverage_PMPS": np.array([True, False]),
            "coverage_SMPS": np.array([True, False]),
            "coverage_HTRU_low": np.array([True, False]),
            "coverage_HTRU_mid": np.array([True, False]),
            "coverage_HTRU_high": np.array([True, False]),
        },
        "expected_update_dictionary_detected": {
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
            "S_radio_obs_mean": np.array([0.001]),
            "S_radio_obs_mean_1400": np.array([0.003]),
            "w_int": np.array([0.05]),
            "w_eff": np.array([0.1]),
            "spectral_index": np.array([-1.8]),
            "idx_det": np.array([0]),
        },
        "expected_update_dictionary_detected_HTRU_low_mid": {
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
            "S_radio_obs_mean": np.array([0.001]),
            "S_radio_obs_mean_1400": np.array([0.003]),
            "w_int": np.array([0.05]),
            "w_eff": np.array([0.1]),
            "spectral_index": np.array([-1.8]),
            "idx_det": np.array([0]),
            "HTRU_low": np.array([True]),
            "HTRU_mid": np.array([True]),
        },
    }

    return data


@pytest.fixture()
def test_case_7():
    data = {
        "expected_header": [
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
        ],
        "expected_units": [
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
        ],
        "expected_header_HTRU_low_mid": [
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
            "HTRU_low",
            "HTRU_mid",
        ],
        "expected_units_HTRU_low_mid": [
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
            "",
            "",
        ],
    }

    return data


def test_initialize_radio_surveys(test_case_2):
    """
    Check that the radio surveys and the dictionaries containing the the detected stars are properly initialized.
    """
    (
        radio_surveys,
        dict_PMPS,
        dict_SMPS,
        dict_HTRU_low_mid,
        dict_HTRU_high,
    ) = sim.initialize_radio_surveys(cfg)

    # Verify the structure of returned dictionaries.
    assert isinstance(radio_surveys, dict)
    assert set(radio_surveys.keys()) == set(
        test_case_2["radio_surveys"].keys()
    )
    for dictionary in [
        dict_PMPS,
        dict_SMPS,
        dict_HTRU_low_mid,
        dict_HTRU_high,
    ]:
        assert isinstance(dictionary, dict)
        assert all(isinstance(value, list) for value in dictionary.values())


def test_load_database_dyn(monkeypatch, test_case_1):
    """
    Check that the dynamical database is correctly imported.
    """

    def mock_select(*args, **kwargs):
        return test_case_1["mock_df_dyn"]

    monkeypatch.setattr(mes, "select", mock_select)

    def mock_polar_to_cartesian(*args, **kwargs):
        return test_case_1["mock_polar_to_cartesian"]

    monkeypatch.setattr(coco, "polar_to_cartesian", mock_polar_to_cartesian)

    def mock_speed_cylindrical_to_cartesian(*args, **kwargs):
        return test_case_1["mock_speed_cylindrical_to_cartesian"]

    monkeypatch.setattr(
        coco,
        "speed_cylindrical_to_cartesian",
        mock_speed_cylindrical_to_cartesian,
    )

    def mock_galactocentric_to_icrs(*args, **kwargs):
        return test_case_1["mock_galactocentric_to_icrs"]

    monkeypatch.setattr(
        coco, "galactocentric_to_icrs", mock_galactocentric_to_icrs
    )

    def mock_galactocentric_to_galactic(*args, **kwargs):
        return test_case_1["mock_galactocentric_to_galactic"]

    monkeypatch.setattr(
        coco, "galactocentric_to_galactic", mock_galactocentric_to_galactic
    )

    dict_out = sim.load_database_dyn(
        test_case_1["dyn_path_pop"],
        test_case_1["n_batchsize"],
        test_case_1["NS_number"],
        test_case_1["idx_remove"],
    )

    assert isinstance(dict_out, dict)
    assert set(dict_out.keys()) == test_case_1["expected_keys"]


def test_apply_surveys_coverage(test_case_2):
    """
    Check that the survey coverage filter is properly applied.
    """
    coverage_database, updated_idx_remove = sim.apply_surveys_coverage(
        test_case_2["radio_surveys"],
        test_case_2["dyn_database_dict"],
        test_case_2["idx_remove"],
        test_case_2["dist_cutoff"],
    )
    assert len(coverage_database["age"]) <= len(
        test_case_2["dyn_database_dict"]["age"]
    )
    assert all(coverage_database["dist"] < test_case_2["dist_cutoff"])
    assert len(updated_idx_remove) == len(
        test_case_2["dyn_database_dict"]["idx"]
    ) - len(coverage_database["idx"])


def test_initialize_population_magrot(test_case_3):
    """
    Check that the dictionary with the initial magneto-rotational properties is properly initialized.
    """
    output_dict = sim.initialize_population_magrot(
        test_case_3["dict_coverage_database"]
    )
    assert set(output_dict.keys()) == test_case_3["expected_keys"]


def test_evolve_population_magrot(monkeypatch, test_case_4, tmp_path):
    """
    Check that the dictionary with the final magneto-rotational properties is properly returned.
    """
    cfg = {
        "a_late": -2.0,
        "save_magrot_evolution": True,  # Enable saving for testing file output.
    }

    output_path = tmp_path

    def mock_magneto_rotational_evolution(*args, **kwargs):
        return (
            test_case_4["B_final"],
            test_case_4["P_final"],
            test_case_4["chi_final"],
            test_case_4["magrot_evol_dict"],
        )

    monkeypatch.setattr(
        mre, "magneto_rotational_evolution", mock_magneto_rotational_evolution
    )

    output_dict = sim.evolve_population_magrot(
        cfg,
        test_case_4["dict_pop_initial_magrot"],
        test_case_4["dict_coverage_database"],
        output_path,
    )

    assert set(output_dict.keys()) == test_case_4["expected_keys"]

    # Check if the file was saved correctly.
    if cfg["save_magrot_evolution"]:
        magrot_file = output_path / "magrot_evolution.json"
        assert magrot_file.exists()


def test_radio_intercepted(monkeypatch, test_case_5):
    """
    Check that the dictionary with the properties of the neutron stars that intercept our line of sight with their
    radio beams is properly returned.
    """

    def mock_calculate_radio_emission(*args, **kwargs):
        return test_case_5["mock_dictionary_intercepted_radio"]

    monkeypatch.setattr(
        er, "calculate_radio_emission", mock_calculate_radio_emission
    )
    out_dict = sim.radio_intercepted(test_case_5["dict_final_pop"])

    # Verify that the keys are correct.
    assert set(out_dict.keys()) == set(test_case_5["expected_keys"])
    # Verify that the output dictionary contains at most the same number of stars as the input one.
    assert len(out_dict["age"]) <= len(
        test_case_5["mock_dictionary_intercepted_radio"]["age"]
    )


def test_radio_detection(test_case_6):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in radio is properly returned.
    """
    (
        dict_PMPS,
        dict_SMPS,
        dict_HTRU_low_mid,
        dict_HTRU_high,
    ) = sim.radio_detection(
        test_case_6["radio_surveys"],
        test_case_6["dictionary_intercepted_radio"],
    )
    # Verify that the keys are correct.
    assert set(dict_PMPS.keys()) == set(
        test_case_6["expected_update_dictionary_detected"].keys()
    )
    assert set(dict_SMPS.keys()) == set(
        test_case_6["expected_update_dictionary_detected"].keys()
    )
    assert set(dict_HTRU_low_mid.keys()) == set(
        test_case_6["expected_update_dictionary_detected_HTRU_low_mid"].keys()
    )
    assert set(dict_HTRU_high.keys()) == set(
        test_case_6["expected_update_dictionary_detected"].keys()
    )

    # Validate overlapping logic for HTRU_low and HTRU_mid.
    assert len(dict_HTRU_low_mid["age"]) == 1
    assert (
        dict_HTRU_low_mid["HTRU_low"]
        == test_case_6["expected_update_dictionary_detected_HTRU_low_mid"][
            "HTRU_low"
        ]
    )
    assert (
        dict_HTRU_low_mid["HTRU_mid"]
        == test_case_6["expected_update_dictionary_detected_HTRU_low_mid"][
            "HTRU_mid"
        ]
    )


def test_create_output_dataframe(test_case_6, test_case_7):
    """
    Check that the dataframe with the output of the survey detection are correct.
    """
    dictionary_detected_PMPS = test_case_6[
        "expected_update_dictionary_detected"
    ].copy()
    dictionary_detected_SMPS = test_case_6[
        "expected_update_dictionary_detected"
    ].copy()
    dictionary_detected_HTRU_low_mid = test_case_6[
        "expected_update_dictionary_detected_HTRU_low_mid"
    ].copy()
    dictionary_detected_HTRU_high = test_case_6[
        "expected_update_dictionary_detected"
    ].copy()

    (
        df_PMPS,
        df_SMPS,
        df_HTRU_low_mid,
        df_HTRU_high,
    ) = sim.create_output_dataframe(
        dictionary_detected_PMPS,
        dictionary_detected_SMPS,
        dictionary_detected_HTRU_low_mid,
        dictionary_detected_HTRU_high,
    )
    # Check MultiIndex columns.
    assert isinstance(df_PMPS.columns, pd.MultiIndex)
    assert isinstance(df_HTRU_low_mid.columns, pd.MultiIndex)

    assert set(df_PMPS.columns.get_level_values(0)) == set(
        test_case_7["expected_header"]
    )
    assert set(df_SMPS.columns.get_level_values(0)) == set(
        test_case_7["expected_header"]
    )
    assert set(df_HTRU_low_mid.columns.get_level_values(0)) == set(
        test_case_7["expected_header_HTRU_low_mid"]
    )
    assert set(df_HTRU_high.columns.get_level_values(0)) == set(
        test_case_7["expected_header"]
    )
    assert set(df_PMPS.columns.get_level_values(1)) == set(
        test_case_7["expected_units"]
    )
    assert set(df_SMPS.columns.get_level_values(1)) == set(
        test_case_7["expected_units"]
    )
    assert set(df_HTRU_low_mid.columns.get_level_values(1)) == set(
        test_case_7["expected_units_HTRU_low_mid"]
    )
    assert set(df_HTRU_high.columns.get_level_values(1)) == set(
        test_case_7["expected_units"]
    )
