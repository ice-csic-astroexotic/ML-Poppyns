"""
Tests for the magneto_rotational_physics/magneto_rotational_evolution_fit module.

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

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.magneto_rotational_physics.magneto_rotational_evolution_fit as mre
from pypopsyn.simulator.configuration import cfg

TOL = 1e-5

# Update the number of simulated objects for testing purposes.
cfg["NS_number"] = 2

# Update the logarithmic time step for testing purposes.
cfg["magrot_time_step_log10"] = 1

# Set to save the time evolution output for testing purposes.
cfg["save_magrot_evolution"] = True


@pytest.fixture()
def test_case_1():
    data = {
        "B_initial": 1e12,
        "t": 1.0e4,
        "B_asymptotic": 1e8,
        "B_expected": 982355579256.808,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "B_initial": 1e12,
        "t": np.array([1.0e4, 1.0e7]),
        "B_asymptotic": 1e8,
        "B_expected": np.array([982355579256.808, 3.31116452e10]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "B_initial": np.array([1e12]),
        "chi_initial": np.array([np.pi / 3]),
        "P_initial": np.array([1.0]),
        "t": 0.0,
        "B_asymptotic": 1e8,
        "dy_expected": np.array([-6.538793128119392e-9, 2.642621780886504e-8]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "B_initial": np.array([1e10, 1e12]),
        "chi_initial": np.array([0.0, np.pi / 3]),
        "P_initial": np.array([1e-2, 1.0]),
        "t_age": np.array([10.0, 10.0]),
        "log_B_asymptotic": np.array([8.0, 8.5]),
        "B_final_expected": np.array([9.99999361e09, 9.99981392e11]),
        "chi_final_expected": np.array([0.0, 1.0471974923]),
        "P_final_expected": np.array([0.01000000136, 1.00000023784]),
        "magrot_evol_dict_expected": {
            0: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
            1: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
        },
    }

    return data


def test_magnetic_field_evolution_fit(test_case_1):
    """
    Testing that the output of the model for the magnetic field evolution is correct.
    """

    B_out = mre.magnetic_field_evolution_fit(
        test_case_1["B_initial"], test_case_1["t"], test_case_1["B_asymptotic"]
    )

    assert np.isclose(
        B_out,
        test_case_1["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_magnetic_field_evolution_fit_numpy(test_case_2):
    """
    Testing that the output of the model for the magnetic field evolution is correct.
    """

    B_out = mre.magnetic_field_evolution_fit_numpy(
        test_case_2["B_initial"], test_case_2["t"], test_case_2["B_asymptotic"]
    )

    assert np.isclose(
        B_out,
        test_case_2["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_combined_derivatives(test_case_3):
    """
    Testing that the output of the combined derivatives has the correct shape.
    """
    y = np.array(
        [
            test_case_3["chi_initial"][0],
            test_case_3["P_initial"][0],
        ]
    )

    dy_out = mre.combined_derivatives(
        0.0, y, test_case_3["B_initial"][0], test_case_3["B_asymptotic"]
    )

    assert np.isclose(
        dy_out,
        test_case_3["dy_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_magneto_rotational_evolution(monkeypatch, test_case_4):
    """
    Verifying (approximately) that the magnetic field, misalignment angle and period are
    correctly evolved in time. To do so, we use a simple finite differencing scheme, i.e.,
    x_initial + x_derivative * time_step, to evaluate the first time step, only, and compare
    it to the output of solve_ivp for two object whose ages correspond to the first evaluated
    time step. With the above choices, the first time_step has a length of 9 years.
    """

    # Mocking the asymptotic magnetic field value.
    def mock_log_B_asymptotic(*args, **kwargs):
        return test_case_4["log_B_asymptotic"]

    monkeypatch.setattr(np.random, "normal", mock_log_B_asymptotic)

    (
        B_final_out,
        chi_final_out,
        P_final_out,
        magrot_evol_dict_out,
    ) = mre.magneto_rotational_evolution(
        test_case_4["B_initial"],
        test_case_4["chi_initial"],
        test_case_4["P_initial"],
        test_case_4["t_age"],
    )
    assert np.isclose(
        B_final_out, test_case_4["B_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()

    assert np.isclose(
        chi_final_out,
        test_case_4["chi_final_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()

    assert np.isclose(
        P_final_out, test_case_4["P_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()

    assert (
        magrot_evol_dict_out.keys()
        == test_case_4["magrot_evol_dict_expected"].keys()
    )

    for key in test_case_4["magrot_evol_dict_expected"].keys():
        assert (
            magrot_evol_dict_out[key].keys()
            == test_case_4["magrot_evol_dict_expected"][key].keys()
        )
