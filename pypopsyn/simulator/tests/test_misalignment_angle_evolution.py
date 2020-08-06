"""
Test for the magneto_rotational_physics/misalignment_angle_evolution module.

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

import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_evolution as mae

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "B": np.array([1e12, 1e13, 1e14, 1e15]),
        "t": np.array([1.0, 1.0, 1.0, 1.0]),
        "chi": np.array([0, np.pi / 3, 1.4 * np.pi, 2.6 * np.pi]),
        "P": np.array([1.0e-2, 1.0e-1, 1.0, 5.0]),
        "chi_deriv_expected": np.array(
            [0.0, -2.073438e-12, -1.407275e-12, 5.629101e-12]
        ),
    }

    return data


def test_misalignment_angle_derivative(test_case_1):
    """
    Verifying that the misalignment angle derivatives for a pulsar sample are evaluated correctly.
    """
    misalignment_angle_derivative_vect = np.vectorize(
        mae.misalignment_angle_derivative
    )
    chi_deriv_out = misalignment_angle_derivative_vect(
        test_case_1["t"],
        test_case_1["chi"],
        test_case_1["B"],
        test_case_1["P"],
    )

    assert np.isclose(
        chi_deriv_out,
        test_case_1["chi_deriv_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
