"""
    Tests for the magnetic_field_derivative module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.magneto_rotational_physics.magnetic_field_evolution as mfev

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
        "t": 1.0e4,
        "B_asymptotic": 1.0e8,
        "sigma": 1e24,
        "L": 1e5,
        "n_e": 1e36,
        "B_expected": 9961884532730.154,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "B_initial": 1e13,
        "t": np.array([1.0e4, 1.0e7]),
        "B_asymptotic": 1.0e8,
        "sigma": 1e24,
        "L": 1e5,
        "n_e": 1e36,
        "B_expected": np.array([9.96188453e12, 6.46395099e11]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "B_initial": 1e12,
        "t": 1.0e4,
        "B_asymptotic": 1e8,
        "a1": -0.13,
        "a2": -3.0,
        "A1": 1.0e14,
        "b1": -0.8,
        "A2": 6.0e8,
        "b2": -0.2,
        "tau_late": 2.0e6,
        "a_late": -2.0,
        "B_expected": 948482242235.077,
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "B_initial": 1e12,
        "t": np.array([1.0e4, 1.0e7]),
        "B_asymptotic": 1e8,
        "a1": -0.13,
        "a2": -3.0,
        "A1": 1.0e14,
        "b1": -0.8,
        "A2": 6.0e8,
        "b2": -0.2,
        "tau_late": 2.0e6,
        "a_late": -2.0,
        "B_expected": np.array([948482242235.077, 1.60959095e10]),
    }

    return data


def test_timescale_ohmic(test_case_1):
    """
    Verifying that the ohmic dissipation timescale is evaluated correctly.
    """
    tau_ohm_out = mfev.timescale_ohmic(test_case_1["L"], test_case_1["sigma"])

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
    tau_Hall_out = mfev.timescale_Hall(
        test_case_1["B"], test_case_1["L"], test_case_1["n_e"]
    )

    assert np.isclose(
        tau_Hall_out,
        test_case_1["tau_Hall_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_magnetic_field_evolution_analytical(test_case_2):
    """
    Verifying that the magnetic field evolution for the analytical model is evaluated correctly.
    """
    B_out = mfev.magnetic_field_evolution_analytical(
        test_case_2["B_initial"],
        test_case_2["t"],
        test_case_2["B_asymptotic"],
        test_case_2["L"],
        test_case_2["sigma"],
        test_case_2["n_e"],
    )

    assert np.isclose(
        B_out,
        test_case_2["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_magnetic_field_evolution_analytical_numpy(test_case_3):
    """
    Verifying that the magnetic field evolution for the analytical model is evaluated correctly.
    """
    B_out = mfev.magnetic_field_evolution_analytical_numpy(
        test_case_3["B_initial"],
        test_case_3["t"],
        test_case_3["B_asymptotic"],
        test_case_3["L"],
        test_case_3["sigma"],
        test_case_3["n_e"],
    )

    assert np.isclose(
        B_out,
        test_case_3["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_magnetic_field_evolution_fit(test_case_4):
    """
    Testing that the output of the model for the magnetic field evolution is correct.
    """

    B_out = mfev.magnetic_field_evolution_fit(
        test_case_4["B_initial"],
        test_case_4["t"],
        test_case_4["B_asymptotic"],
        test_case_4["a1"],
        test_case_4["a2"],
        test_case_4["A1"],
        test_case_4["A2"],
        test_case_4["b1"],
        test_case_4["b2"],
        test_case_4["tau_late"],
        test_case_4["a_late"],
    )

    assert np.isclose(
        B_out,
        test_case_4["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_magnetic_field_evolution_fit_numpy(test_case_5):
    """
    Testing that the output of the model for the magnetic field evolution is correct.
    """

    B_out = mfev.magnetic_field_evolution_fit_numpy(
        test_case_5["B_initial"],
        test_case_5["t"],
        test_case_5["B_asymptotic"],
        test_case_5["a1"],
        test_case_5["a2"],
        test_case_5["A1"],
        test_case_5["A2"],
        test_case_5["b1"],
        test_case_5["b2"],
        test_case_5["tau_late"],
        test_case_5["a_late"],
    )

    assert np.isclose(
        B_out,
        test_case_5["B_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
