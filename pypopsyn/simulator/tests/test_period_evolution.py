"""
Test for the magneto_rotational_physics/period_evolution module.

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

import pypopsyn.simulator.magneto_rotational_physics.period_evolution as pe

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "B": np.array([1e12, 1e13, 1e14, 1e15]),
        "chi": np.array([0, np.pi / 2, np.pi, 2 * np.pi]),
        "P": np.array([1.0e-3, 1.0e-1, 1, 5]),
        "P_deriv_expected": np.array(
            [4.788399e-13, 9.576799e-13, 4.788399e-12, 9.576799e-11]
        ),
    }

    return data


def test_period_derivative(test_case_1):
    """
    Verifying that the period derivatives for a pulsar sample are evaluated correctly.
    """
    P_deriv_out = pe.period_derivative(
        test_case_1["B"], test_case_1["chi"], test_case_1["P"]
    )

    assert np.isclose(
        P_deriv_out, test_case_1["P_deriv_expected"], rtol=TOL, atol=1.0e9,
    ).all()
