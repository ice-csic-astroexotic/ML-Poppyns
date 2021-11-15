"""
Tests for electron density model module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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

    tau_sc_out = edm.compute_tau_sc(test_case_1["DM"], test_case_1["nu"],)

    assert np.isclose(
        test_case_1["tau_sc_expected"], tau_sc_out, rtol=TOL, atol=1.0e-5
    ).all()
