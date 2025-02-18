"""
    Tests for the survey_x module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import pypopsyn.simulator.multiband_emission.emission_xray as ex
import pypopsyn.simulator.multiband_surveys.survey_x as sx


@pytest.fixture()
def test_case_1():
    data = {
        "P": np.array([0.5, 1.0, 2.0, 1.5, 3.0]),
        "B": np.array([1e12, 2e12, 3e12, 4e12, 5e12]),
        "B_initial": np.array([1e14, 2e14, 3e14, 4e14, 5e14]),
        "chi": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "age": np.array([1e4, 2e4, 3e4, 4e4, 5e4]),
        "ra": np.array([10, 20, 30, 40, 50]),
        "dec": np.array([-10, -20, -30, -40, -50]),
        "dist": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "coverage": np.array([True, False, True, False, True]),
        "L_x_threshold": 1e29,
        "S_x_abs_threshold": 1e-15,
        "dummy_L_x_interpolator": RectBivariateSpline(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        ),
        "mock_S_x_abs": np.array(
            [1.0e-15, 2.0e-14, 3.0e-12, 4.0e-15, 5.0e-14]
        ),
        "mock_N_H": np.array([1.0e-21, 2.0e-21, 3.0e-21, 4.0e-21, 5.0e-21]),
        "detected_x_expected": np.array([False, False, True, False, True]),
    }

    return data


def test_detected_x_population(test_case_1, monkeypatch):
    def mock_interpolator(*args, **kwargs):
        L_x = 1.0e34 * np.ones(len(test_case_1["B"]))
        return L_x

    monkeypatch.setattr(RectBivariateSpline, "ev", mock_interpolator)

    def mock_flux_xray_absorbed(*args, **kwargs):
        S_x_abs = test_case_1["mock_S_x_abs"][test_case_1["coverage"]]
        N_H = test_case_1["mock_N_H"][test_case_1["coverage"]]
        return S_x_abs, N_H

    monkeypatch.setattr(ex, "flux_xray_absorbed", mock_flux_xray_absorbed)

    # Run the function.
    detected_x, L_x_therm, S_x_abs, N_H, P_dot = sx.detected_x_population(
        test_case_1["P"],
        test_case_1["B"],
        test_case_1["B_initial"],
        test_case_1["chi"],
        test_case_1["age"],
        test_case_1["ra"],
        test_case_1["dec"],
        test_case_1["dist"],
        test_case_1["coverage"],
        test_case_1["dummy_L_x_interpolator"],
        test_case_1["L_x_threshold"],
        test_case_1["S_x_abs_threshold"],
    )
    print(S_x_abs)

    # Assertions
    assert detected_x.shape == (
        len(test_case_1["P"]),
    ), "Output detected_x shape mismatch"
    assert np.all(
        L_x_therm[test_case_1["coverage"]] > test_case_1["L_x_threshold"]
    ), "Luminosity threshold filtering incorrect"
    assert np.all(
        S_x_abs[test_case_1["detected_x_expected"]]
        > test_case_1["S_x_abs_threshold"]
    ), "Flux filtering incorrect"
    assert np.all(detected_x) == np.all(test_case_1["detected_x_expected"])
