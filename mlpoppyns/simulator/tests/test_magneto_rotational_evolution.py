"""
    Tests for the magneto_rotational_evolution_fit module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import mlpoppyns.simulator.basics.constants as const
import mlpoppyns.simulator.magneto_rotational_physics.magneto_rotational_evolution as mre
from mlpoppyns.simulator.config_simulator import cfg

TOL = 1e-5

# Update the number of simulated objects for testing purposes.
cfg["NS_number"] = 2

# Update the neutron star radius in [cm] for testing purposes.
cfg["NS_radius"] = 1.1e6

# Update neutron star mass in solar masses for testing purposes.
cfg["NS_mass"] = 1.4 * const.M_SUN

# Update the logarithmic time step for testing purposes.
cfg["magrot_time_step_log10"] = 1

# Set to save the time evolution output for testing purposes.
cfg["save_magrot_evolution"] = True

# Update the conductivity coefficient for testing purposes.
cfg["sigma"] = 1e24

# Update the characteristic length scale of the magnetic field in [cm] for testing purposes.
cfg["L"] = 1e5

# Update the characteristic electron density in [g/cm^3] for testing purposes.
cfg["n_e"] = 1e36

# Set the fit parameters for the magnetic field evolution for testing purposes.
cfg["a1"] = -0.13
cfg["a2"] = -3.0
cfg["A1"] = 1.0e14
cfg["b1"] = -0.8
cfg["A2"] = 6.0e8
cfg["b2"] = -0.2
cfg["tau_late"] = 2.0e6
cfg["a_late"] = -2.0


@pytest.fixture()
def test_case_1():
    data = {
        "B_initial": np.array([1e12]),
        "chi_initial": np.array([np.pi / 3]),
        "P_initial": np.array([1.0]),
        "t": 0.0,
        "B_asymptotic": 1e8,
        "sigma": 1e24,
        "L": 1e5,
        "n_e": 1e36,
        "dy_expected": np.array([-6.538793128119392e-9, 2.642621780886504e-8]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "B_initial": np.array([1e12]),
        "chi_initial": np.array([np.pi / 3]),
        "P_initial": np.array([1.0]),
        "t": 0.0,
        "B_asymptotic": 1e8,
        "a1": -0.13,
        "a2": -3.0,
        "A1": 1.0e14,
        "b1": -0.8,
        "A2": 6.0e8,
        "b2": -0.2,
        "tau_late": 2.0e6,
        "a_late": -2.0,
        "dy_expected": np.array([-6.538793128119392e-9, 2.642621780886504e-8]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "B_initial": np.array([1e10, 1e12]),
        "chi_initial": np.array([0.0, np.pi / 3]),
        "P_initial": np.array([1e-2, 1.0]),
        "t_age": np.array([10.0, 10.0]),
        "log_B_asymptotic": np.array([8.0, 8.5]),
        "B_final_expected": np.array([9.99997743e09, 9.99997588e11]),
        "chi_final_expected": np.array([0.0, 1.0471974923]),
        "P_final_expected": np.array([0.01000000136, 1.00000023784]),
        "magrot_evol_dict_expected": {
            0: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
            1: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
        },
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
        "B_final_expected": np.array([9.99989350e9, 9.99938908e11]),
        "chi_final_expected": np.array([0.0, 1.0471974923]),
        "P_final_expected": np.array([0.01000000136, 1.00000023784]),
        "magrot_evol_dict_expected": {
            0: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
            1: {"t": None, "B(t)": None, "chi(t)": None, "P(t)": None},
        },
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "age": np.array([1e6, 2e6]),
        "expected_keys": {
            "age",
            "B",
            "P",
            "P_dot",
            "chi",
        },
    }

    return data


@pytest.fixture()
def test_case_6():
    data = {
        "dict_pop_initial_magrot": {
            "age": np.array([1e6]),
            "B": np.array([1.0e12]),
            "P": np.array([0.1]),
            "chi": np.array([0.5]),
        },
        "B_final": np.array([1.0e12]),
        "P_final": np.array([0.1]),
        "chi_final": np.array([0.5]),
        "magrot_evol_dict": {
            "0": {
                "t": [],
                "B(t)": [],
                "P(t)": [],
                "chi(t)": [],
            }
        },
        "expected_keys": {
            "B_initial",
            "B",
            "P",
            "P_dot",
            "chi",
        },
    }

    return data


def test_combined_derivatives_analytical(test_case_1):
    """
    Testing that the output of the combined derivatives has the correct shape.
    """
    y = np.array(
        [
            test_case_1["chi_initial"][0],
            test_case_1["P_initial"][0],
        ]
    )

    dy_out = mre.combined_derivatives_analytical(
        0.0,
        y,
        test_case_1["B_initial"][0],
        test_case_1["B_asymptotic"],
        test_case_1["L"],
        test_case_1["sigma"],
        test_case_1["n_e"],
        cfg["NS_mass"],
        cfg["NS_radius"],
    )

    assert np.isclose(
        dy_out,
        test_case_1["dy_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_combined_derivatives_fit(test_case_2):
    """
    Testing that the output of the combined derivatives has the correct shape.
    """
    y = np.array(
        [
            test_case_2["chi_initial"][0],
            test_case_2["P_initial"][0],
        ]
    )

    dy_out = mre.combined_derivatives_fit(
        0.0,
        y,
        test_case_2["B_initial"][0],
        test_case_2["B_asymptotic"],
        test_case_2["a1"],
        test_case_2["a2"],
        test_case_2["A1"],
        test_case_2["A2"],
        test_case_2["b1"],
        test_case_2["b2"],
        test_case_2["tau_late"],
        test_case_2["a_late"],
        cfg["NS_mass"],
        cfg["NS_radius"],
    )

    assert np.isclose(
        dy_out,
        test_case_2["dy_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_magneto_rotational_evolution_analytical(monkeypatch, test_case_3):
    """
    Verifying (approximately) that the magnetic field, misalignment angle and period are
    correctly evolved in time for the analytical model. To do so, we use a simple finite differencing scheme, i.e.,
    x_initial + x_derivative * time_step, to evaluate the first time step, only, and compare
    it to the output of solve_ivp for two object whose ages correspond to the first evaluated
    time step. With the above choices, the first time_step has a length of 9 years.
    """

    # Mocking the asymptotic magnetic field value.
    def mock_log_B_asymptotic(*args, **kwargs):
        return test_case_3["log_B_asymptotic"]

    monkeypatch.setattr(np.random, "normal", mock_log_B_asymptotic)

    cfg["magneto-thermal_model"] = "analytical"

    (
        B_final_out,
        chi_final_out,
        P_final_out,
        magrot_evol_dict_out,
    ) = mre.magneto_rotational_evolution(
        test_case_3["B_initial"],
        test_case_3["chi_initial"],
        test_case_3["P_initial"],
        test_case_3["t_age"],
    )
    assert np.isclose(
        B_final_out, test_case_3["B_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()

    assert np.isclose(
        chi_final_out,
        test_case_3["chi_final_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()

    assert np.isclose(
        P_final_out, test_case_3["P_final_expected"], rtol=TOL, atol=1.0e-30
    ).all()

    assert (
        magrot_evol_dict_out.keys()
        == test_case_3["magrot_evol_dict_expected"].keys()
    )

    for key in test_case_3["magrot_evol_dict_expected"].keys():
        assert (
            magrot_evol_dict_out[key].keys()
            == test_case_3["magrot_evol_dict_expected"][key].keys()
        )


def test_magneto_rotational_evolution_fit(monkeypatch, test_case_4):
    """
    Verifying (approximately) that the magnetic field, misalignment angle and period are
    correctly evolved in time for a fit model. To do so, we use a simple finite differencing scheme, i.e.,
    x_initial + x_derivative * time_step, to evaluate the first time step, only, and compare
    it to the output of solve_ivp for two object whose ages correspond to the first evaluated
    time step. With the above choices, the first time_step has a length of 9 years.
    """

    # Mocking the asymptotic magnetic field value.
    def mock_log_B_asymptotic(*args, **kwargs):
        return test_case_4["log_B_asymptotic"]

    monkeypatch.setattr(np.random, "normal", mock_log_B_asymptotic)

    cfg["magneto-thermal_model"] = "SLy4_dip-tor_heavy"

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


def test_initialize_population_magrot(test_case_5):
    """
    Check that the dictionary with the initial magneto-rotational properties is properly initialized.
    """
    output_dict = mre.initialize_population_magrot(test_case_5["age"])
    assert set(output_dict.keys()) == test_case_5["expected_keys"]


def test_evolve_population_magrot(monkeypatch, test_case_6, tmp_path):
    """
    Check that the dictionary with the final magneto-rotational properties is properly returned.
    """
    cfg[
        "save_magrot_evolution"
    ] = True  # Enable saving for testing file output.

    output_path = tmp_path

    def mock_magneto_rotational_evolution(*args, **kwargs):
        return (
            test_case_6["B_final"],
            test_case_6["P_final"],
            test_case_6["chi_final"],
            test_case_6["magrot_evol_dict"],
        )

    monkeypatch.setattr(
        mre, "magneto_rotational_evolution", mock_magneto_rotational_evolution
    )

    output_dict = mre.evolve_population_magrot(
        test_case_6["dict_pop_initial_magrot"],
        output_path,
    )

    assert set(output_dict.keys()) == test_case_6["expected_keys"]

    # Check if the file was saved correctly.
    if cfg["save_magrot_evolution"]:
        magrot_file = output_path / "magrot_evolution.json"
        assert magrot_file.exists()
