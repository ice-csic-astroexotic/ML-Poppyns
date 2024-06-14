"""
    Tests for e_density_model module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.interstellar_medium.e_density_model as edm

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "DM": np.array([100, 1000]),
        "nu": 1.4e9,
        "mock_log10_tau_sc": np.array([-2, 0.9]),
        "tau_sc_expected": np.array([1.66359e-05, 1.32144e-02]),
    }

    return data


def test_compute_tau_sc(monkeypatch, test_case_1):
    """
    Verifying that for a given choice of the DM and survey frequency the scattering timescale is computed correctly.
    """

    # Mocking the parameter that is otherwise randomly determined.
    def mock_random_normal(*args, **kwargs):
        return test_case_1["mock_log10_tau_sc"]

    monkeypatch.setattr(np.random, "normal", mock_random_normal)

    tau_sc_out = edm.compute_tau_sc(
        test_case_1["DM"],
        test_case_1["nu"],
    )

    assert np.isclose(
        test_case_1["tau_sc_expected"], tau_sc_out, rtol=TOL, atol=1.0e-5
    ).all()
