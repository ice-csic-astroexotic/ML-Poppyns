"""
    Tests for nh_model module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import mlpoppyns.simulator.interstellar_medium.nh_model as nhm

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "RA": np.array([60.0, 120.0, 250.0]),
        "DEC": np.array([35.0, 45.0, 55.0]),
        "d": np.array([5.0, 10.0, 15.0]),
        "DM": np.array([100.0, 200.0, 300.0]),
        "expected_NH": np.array([2.38085937e21, 5.48339844e20, 2.89306641e20]),
        "expected_NH_from_DM": np.array([3.0e21, 6.0e21, 9.0e21]),
    }

    return data


def test_compute_NH(test_case_1):
    """
    Test if the N_H value is correctly computed from the map of Doroshenko (2024).
    """
    computed_NH = nhm.compute_NH(
        test_case_1["RA"], test_case_1["DEC"], test_case_1["d"]
    )
    assert np.isclose(
        test_case_1["expected_NH"], computed_NH, rtol=TOL, atol=1.0e-5
    ).all()


def test_compute_NH_from_DM(test_case_1):
    """
    Test if the N_H value is correctly computed from the DM-N_H relation of He, Ng and Kaspi (2013).
    """
    computed_NH = nhm.compute_NH_from_DM(test_case_1["DM"])
    assert np.isclose(
        test_case_1["expected_NH_from_DM"], computed_NH, rtol=TOL, atol=1.0e-5
    ).all()
