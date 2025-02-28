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
        "S_x_abs_threshold": 1e-15,
        "S_x_abs": np.array([1.0e-16, 2.0e-16, 3.0e-12, 4.0e-16, 5.0e-14]),
        "detected_x_expected": np.array([False, False, True, False, True]),
    }

    return data


def test_detected_x_population(test_case_1, monkeypatch):
    """
    Verifying that the X-ray detection with a flux threshold is computed correctly.
    """

    # Run the function.
    detected_x = sx.detected_x_population_flux_threshold(
        test_case_1["S_x_abs"],
        test_case_1["S_x_abs_threshold"],
    )

    assert np.all(detected_x) == np.all(test_case_1["detected_x_expected"])
