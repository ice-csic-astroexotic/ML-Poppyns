"""
Tests for the magneto_rotational_physics/magnetic_field_derivative module.

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

import pypopsyn.simulator.magneto_rotational_physics.magnetic_field_derivative as mfdv

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "sigma": 1e24,
        "L": 1e5,
        "n_e": 1e36,
        "B": 1e12,
        "tau_ohm_expected": 4.43365e6,
        "tau_Hall_expected": 63.8430471e6,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "B_initial": 1e13,
        "B": 5e12,
        "B_deriv_expected": -5043591.01648,
    }

    return data


def test_timescale_ohmic(test_case_1):
    """
    Verifying that the ohmic dissipation timescale is evaluated correctly.
    """
    tau_ohm_out = mfdv.timescale_ohmic(test_case_1["L"], test_case_1["sigma"])

    assert np.isclose(
        tau_ohm_out,
        test_case_1["tau_ohm_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_timescale_Hall(test_case_1):
    """
    Verifying that the Hall timescale is evaluated correctly.
    """
    tau_Hall_out = mfdv.timescale_Hall(
        test_case_1["B"], test_case_1["L"], test_case_1["n_e"]
    )

    assert np.isclose(
        tau_Hall_out,
        test_case_1["tau_Hall_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_field_derivative(test_case_2):
    """
    Verifying that the magnetic field derivative is evaluated correctly.
    """
    B_deriv_out = mfdv.field_derivative(
        test_case_2["B"], test_case_2["B_initial"]
    )

    assert np.isclose(
        B_deriv_out,
        test_case_2["B_deriv_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
