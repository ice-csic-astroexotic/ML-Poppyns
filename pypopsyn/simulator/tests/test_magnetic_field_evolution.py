"""
Tests for the magnetic_field_evolution module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)

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

import pypopsyn.simulator.magnetic_field_evolution as mfe

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "sigma": 1e24,
        "L": 1e5,
        "n_e": 1e36,
        "B": np.array([1e12, 1e14]),
        "tau_ohm_expected": 4433654.543524,
        "tau_Hall_expected": np.array([63843047.417426, 638430.474174]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "B_initial": np.array([1e12, 1e13, 1e14]),
        "B": np.array([1e11, 5e12, 1e10]),
        "B_deriv_expected": np.array(
            [-0.720173e-05, -4817.742130e-05, -7.157033e-05]
        ),
    }

    return data


def test_timescale_ohmic(test_case_1):
    """
    Verifying that the ohmic dissipation timescale is evaluated correctly.
    """
    tau_ohm_out = mfe.timescale_ohmic(test_case_1["L"], test_case_1["sigma"])

    assert np.isclose(
        tau_ohm_out, test_case_1["tau_ohm_expected"], rtol=TOL, atol=1.0e9,
    )


def test_timescale_Hall(test_case_1):
    """
    Verifying that the Hall timescale is evaluated correctly.
    """
    tau_Hall_out = mfe.timescale_Hall(
        test_case_1["B"], test_case_1["L"], test_case_1["n_e"]
    )

    assert np.isclose(
        tau_Hall_out, test_case_1["tau_Hall_expected"], rtol=TOL, atol=1.0e9,
    ).all()


def test_magnetic_field_derivative(test_case_2):
    """
    Verifying that the magnetic field derivatives are evaluated correctly.
    """
    B_deriv_out = mfe.field_derivative(
        test_case_2["B"], test_case_2["B_initial"]
    )

    assert np.isclose(
        B_deriv_out, test_case_2["B_deriv_expected"], rtol=TOL, atol=1.0e9,
    ).all()
