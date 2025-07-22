"""
    Tests for the survey_x module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import json
import tempfile

import numpy as np
import pytest

import pypopsyn.simulator.multiband_surveys.survey_xray as sx

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "S_x_threshold": 1e-15,
        "S_x": np.array([1.0e-16, 2.0e-16, 3.0e-12, 4.0e-16, 5.0e-14]),
        "detected_x_expected": np.array([False, False, True, False, True]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "S_x_threshold_log10_mean": -14,
        "S_x_threshold_log10_sigma": 0.5,
        "S_x": np.array([1.0e-16, 2.0e-16, 3.0e-12, 4.0e-16, 5.0e-14]),
        "detected_x_expected": np.array([False, False, True, False, False]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "dummy_params_flux_threshold": {
            "apply_sharp_flux_threshold": True,
            "sharp_flux_threshold": 1e-14,
            "S_x_threshold_log10_mean_long_exposure": -14.0,
            "S_x_threshold_log10_sigma_long_exposure": 0.5,
            "S_x_threshold_log10_mean_short_exposure": -13.0,
            "S_x_threshold_log10_sigma_short_exposure": 0.5,
            "RA_range": [100.0, 200.0],
            "DEC_range": [-30.0, 30.0],
            "l_range": [-50.0, 150.0],
            "b_range_abs": [-5.0, 20.0],
            "name": "Test Survey",
        },
        "dummy_params_realistic": {
            "apply_sharp_flux_threshold": False,
            "sharp_flux_threshold": 1e-14,
            "S_x_threshold_log10_mean_long_exposure": -14.0,
            "S_x_threshold_log10_sigma_long_exposure": 0.5,
            "S_x_threshold_log10_mean_short_exposure": -13.0,
            "S_x_threshold_log10_sigma_short_exposure": 0.5,
            "RA_range": [100.0, 200.0],
            "DEC_range": [-30.0, 30.0],
            "l_range": [-50.0, 150.0],
            "b_range_abs": [-5.0, 20.0],
            "name": "Test Survey",
        },
        "l_gal": np.array([-90.0, 0.0]),
        "b_gal": np.array([0.0, 20.0]),
        "RA": np.array([30.0, 150.0]),
        "DEC": np.array([-20.0, 10.0]),
        "coverage_expected": np.array([False, True], dtype=bool),
        "S_x": np.array([1.0e-15, 1.0e-11]),
        "outburst": np.array([False, True], dtype=bool),
        "detected_flux_threshold_expected": np.array(
            [False, True], dtype=bool
        ),
        "detected_realistic_expected": np.array([False, True], dtype=bool),
    }

    return data


def test_sharp_flux_filter(test_case_1):
    """
    Verifying that the X-ray detection with a flux threshold is computed correctly.
    """

    # Run the function.
    detected_x = sx.sharp_flux_filter(
        test_case_1["S_x"],
        test_case_1["S_x_threshold"],
    )

    assert np.all(detected_x) == np.all(test_case_1["detected_x_expected"])


def test_smooth_flux_filter(test_case_2, monkeypatch):
    """
    Verifying that the X-ray detection with a flux threshold is computed correctly.
    """

    # Mocking the flux threshold from a gaussian distribution in log10.
    def mock_random_normal(*args, **kwargs):
        mocked_rand = np.array([-14.0, -14.1, -13.9, -14.0, -13.9])
        return mocked_rand

    monkeypatch.setattr(np.random, "normal", mock_random_normal)

    # Run the function.
    detected_x = sx.smooth_flux_filter(
        test_case_2["S_x"],
        test_case_2["S_x_threshold_log10_mean"],
        test_case_2["S_x_threshold_log10_sigma"],
    )

    assert np.all(detected_x) == np.all(test_case_2["detected_x_expected"])


def test_sky_coverage(test_case_3):
    """
    Verifying that the sky coverage of a survey is computed correctly.
    """
    # Create a temporary .json file with some survey parameters to initialize an X-ray survey class object.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmpfile:
        json.dump(test_case_3["dummy_params_flux_threshold"], tmpfile)
        tmpfile_path = tmpfile.name

    survey = sx.SurveyXray(parameters_path=tmpfile_path)

    coverage_out = survey.sky_coverage(
        test_case_3["RA"],
        test_case_3["DEC"],
        test_case_3["l_gal"],
        test_case_3["b_gal"],
    )

    assert test_case_3["coverage_expected"].all() == coverage_out.all()


def test_detect_xray_population(test_case_3, monkeypatch):
    """
    Verifying that a population of pulsars is correctly detected by the survey.
    """

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmpfile:
        json.dump(test_case_3["dummy_params_flux_threshold"], tmpfile)
        tmpfile_path = tmpfile.name

    survey_flux_threshold = sx.SurveyXray(parameters_path=tmpfile_path)

    detected_out = survey_flux_threshold.detected_xray_population(
        test_case_3["S_x"],
        test_case_3["outburst"],
    )

    assert (
        test_case_3["detected_flux_threshold_expected"].all()
        == detected_out.all()
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmpfile:
        json.dump(test_case_3["dummy_params_realistic"], tmpfile)
        tmpfile_path = tmpfile.name

    survey_flux_threshold = sx.SurveyXray(parameters_path=tmpfile_path)

    # Mocking the flux threshold from a gaussian distribution in log10.
    def mock_smooth_flux_filter(*args, **kwargs):
        mocked_rand = np.array([False, True])
        return mocked_rand

    monkeypatch.setattr(sx, "smooth_flux_filter", mock_smooth_flux_filter)

    detected_out = survey_flux_threshold.detected_xray_population(
        test_case_3["S_x"],
        test_case_3["outburst"],
    )

    assert (
        test_case_3["detected_realistic_expected"].all() == detected_out.all()
    )
