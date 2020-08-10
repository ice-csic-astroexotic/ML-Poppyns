"""
Tests for the magneto_rotational_physics/magneto_rotational_evolution module.

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

import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
from pypopsyn.simulator.configuration import cfg

TOL = 1e-10

# Update the number of simulated objects for testing purposes.
cfg["NS_number"] = 2

# Update the logarithmic time step for testing purposes.
cfg["time_step_log10"] = 1


@pytest.fixture()
def test_case_1():
    data = {
        "B_initial": np.array([1e10, 1e12]),
        "chi_initial": np.array([0.0, np.pi / 3]),
        "P_initial": np.array([1e-2, 1.0]),
        "t_age": np.array([10, 10]),
        "B_final_expected": np.array([9.999979686e9, 999.997829102e9]),
        "chi_final_expected": np.array([0.0, 1.0471974923]),
        "P_final_expected": np.array([0.01000000136, 1.00000023784]),
    }

    return data


def test_combined_derivatives(test_case_1):
    """
    Testing that the output of the combined derivatives has the correct shape.
    """
    y = np.array(
        [
            test_case_1["B_initial"][0],
            test_case_1["chi_initial"][0],
            test_case_1["P_initial"][0],
        ]
    )

    dy = mre.combined_derivatives(0, y, test_case_1["B_initial"][0])
    assert len(dy) == 3


def test_magneto_rotational_evolution(test_case_1):
    """
    Verifying (approximately) that the magnetic field, misalignment angle and period are
    correctly evolved in time. To do so, we use a simple finite differencing scheme, i.e.,
    x_initial + x_derivative * time_step, to evaluate the first time step, only, and compare
    it to the output of solve_ivp for two object whose ages correspond to the first evaluated
    time step. With the above choices, the first time_step has a length of 9 years.
    """
    B_final_out, chi_final_out, P_final_out = mre.magneto_rotational_evolution(
        test_case_1["B_initial"],
        test_case_1["chi_initial"],
        test_case_1["P_initial"],
        test_case_1["t_age"],
    )
    assert np.isclose(
        B_final_out, test_case_1["B_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()

    assert np.isclose(
        chi_final_out,
        test_case_1["chi_final_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()

    assert np.isclose(
        P_final_out, test_case_1["P_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()
