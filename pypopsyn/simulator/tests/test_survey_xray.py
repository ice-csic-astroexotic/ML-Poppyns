"""
    Tests for the survey_x module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import pypopsyn.simulator.multiband_emission.emission_xray as ex
import pypopsyn.simulator.multiband_surveys.survey_xray as sx


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


def test_detected_x_population_sharp_flux_filter(test_case_1):
    """
    Verifying that the X-ray detection with a flux threshold is computed correctly.
    """

    # Run the function.
    detected_x = sx.detected_x_population_sharp_flux_filter(
        test_case_1["S_x"],
        test_case_1["S_x_threshold"],
    )

    assert np.all(detected_x) == np.all(test_case_1["detected_x_expected"])


def test_detected_x_population_smooth_flux_filter(test_case_2, monkeypatch):
    """
    Verifying that the X-ray detection with a flux threshold is computed correctly.
    """

    # Mocking the flux threshold from a gaussian distribution in log10.
    def mock_random_normal(*args, **kwargs):
        mocked_rand = np.array([-14.0, -14.1, -13.9, -14.0, -13.9])
        return mocked_rand

    monkeypatch.setattr(np.random, "normal", mock_random_normal)

    # Run the function.
    detected_x = sx.detected_x_population_smooth_flux_filter(
        test_case_2["S_x"],
        test_case_2["S_x_threshold_log10_mean"],
        test_case_2["S_x_threshold_log10_sigma"],
    )

    assert np.all(detected_x) == np.all(test_case_2["detected_x_expected"])
