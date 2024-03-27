"""
Tests for N_H model module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2024

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

import pypopsyn.simulator.interstellar_medium.nh_model as nhm

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
