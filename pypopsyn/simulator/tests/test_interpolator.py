"""
Tests for the basics/interpolator module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

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

import pypopsyn.simulator.basics.interpolator as itp

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "x_in": np.array([0.1, 0.3, 0.5]),
        "y_in": np.array([0.1, 0.3, 0.5]),
        "f_in": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "x_out": np.array([0.2, 0.4]),
        "y_out": np.array([0.2, 0.4]),
        "f_out_expected": np.array([[0.25, 0.25], [0.25, 0.25]]),
    }

    return data


def test_bilinear_interpolation(test_case_1):
    """
    Checking that the interpolation is correctly calculated for given table of data.
    """
    f_out = itp.bilinear_interpolation(
        test_case_1["x_in"],
        test_case_1["y_in"],
        test_case_1["f_in"],
        test_case_1["x_out"],
        test_case_1["y_out"],
    )
    assert np.isclose(f_out, test_case_1["f_out_expected"]).all()
