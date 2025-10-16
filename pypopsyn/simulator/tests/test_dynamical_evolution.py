"""
    Tests for the dynamical_evolution module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import pypopsyn.simulator.initial_population as ipop
import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
import pypopsyn.simulator.stellar_dynamics.galactic_model as gm
from pypopsyn.simulator.config_simulator import cfg

gm.initialize_galactic_model()

TOL = 1e-5

# Select the galactic model from Marchetti et al. (2019) for the test.
cfg["galactic_model"] = "gmM19"

# Update the time step for testing purposes.
cfg["dyn_time_step"] = 1.0e4

# Set to save the time evolution output for testing purposes.
cfg["save_dyn_evolution"] = True


class MockInitialNeutronStarPopulation:
    def __init__(self, NS_number):
        self.NS_number = NS_number

    def position(self, t_age):
        return (
            np.array([1.0, 2.0, 3.0]),  # r_initial
            np.array([0.1, 0.2, 0.3]),  # phi_initial
            np.array([0.01, 0.02, 0.03]),  # z_initial
        )

    def kick_velocity(self):
        return (
            np.array([100, 200, 300]),  # vk_r
            np.array([150, 250, 350]),  # vk_phi
            np.array([50, 60, 70]),  # vk_z
        )

    def orbital_velocity(self, r, z):
        return np.array([10, 20, 30])  # v_orb


# Mock galactic_model methods.
class MockGalacticModel:
    def total_energy(self, v, r, z):
        return 1e50

    def total_angular_momentum_z(self, v_phi, r):
        return 1e45


@pytest.fixture()
def test_case_1():

    data = {
        "t": 0.0,
        "initial_cond": np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "derivatives_expected": np.array(
            [0.0, 0.0, 0.0, -1.26283e-14, 0.0, -2.49464e-14]
        ),
        "galactic_model": gm.galactic_model,
    }

    return data


@pytest.fixture()
def test_case_2():

    data = {
        "initial_cond": np.array(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
        ),
        "t_age": np.array([1.0e4, 1.0e4]),
        "final_population_expected": np.array(
            [
                [1.0, 0.0, 1.0, -1.26283e-10, 0.0, -2.49464e-10],
                [1.0, 0.0, -1.0, -1.26283e-10, 0.0, 2.49464e-10],
            ]
        ),
        "dyn_evol_dict_expected": {
            0: {
                "t": None,
                "r(t)": None,
                "phi(t)": None,
                "z(t)": None,
                "v_r(t)": None,
                "v_phi(t)": None,
                "v_z(t)": None,
            },
            1: {
                "t": None,
                "r(t)": None,
                "phi(t)": None,
                "z(t)": None,
                "v_r(t)": None,
                "v_phi(t)": None,
                "v_z(t)": None,
            },
        },
    }

    return data


@pytest.fixture()
def test_case_3():

    data = {
        "age": np.array([1e6, 2e6, 3e6]),
        "dict_expected": {
            "age": np.array([1e6, 2e6, 3e6]),
            "r": np.array([1.0, 2.0, 3.0]),
            "phi": np.array([0.1, 0.2, 0.3]),
            "z": np.array([0.01, 0.02, 0.03]),
            "v_r": np.array([100, 200, 300]),
            "v_phi": np.array([160, 270, 380]),
            "v_z": np.array([50, 60, 70]),
            "v_orb": np.array([10, 20, 30]),
        },
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "dict_pop_initial_dyn": {
            "age": np.array([1e6]),
            "r": np.array([1.0]),
            "phi": np.array([0.1]),
            "z": np.array([0.01]),
            "v_r": np.array([100]),
            "v_phi": np.array([160]),
            "v_z": np.array([50]),
            "v_orb": np.array([10]),
        },
        "mock_final_population": np.array(
            [
                np.array([1.0]),
                np.array([0.1]),
                np.array([0.01]),
                np.array([100]),
                np.array([160]),
                np.array([50]),
            ]
        ).T,
        "dyn_evol_dict": {
            "0": {
                "t": [],
                "r(t)": [],
                "phi(t)": [],
                "z(t)": [],
                "v_r(t)": [],
                "v_phi(t)": [],
                "v_z(t)": [],
            }
        },
        "expected_keys": {
            "age",
            "r",
            "phi",
            "z",
            "v_r",
            "v_phi",
            "v_z",
        },
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "dict_initial": {
            "r": np.array([1.0, 2.0]),
            "z": np.array([0.1, 0.2]),
            "v_r": np.array([10.0, 20.0]),
            "v_phi": np.array([30.0, 40.0]),
            "v_z": np.array([50.0, 60.0]),
        },
        "dict_final": {
            "r": np.array([1.1, 2.1]),
            "z": np.array([0.15, 0.25]),
            "v_r": np.array([11.0, 21.0]),
            "v_phi": np.array([31.0, 41.0]),
            "v_z": np.array([51.0, 61.0]),
        },
    }

    return data


def test_dynamical_eq_system(test_case_1):
    """
    Verifying that the dynamical equation system evaluates the derivatives correctly.
    """
    derivatives_out = dyn.dynamical_eq_system(
        test_case_1["t"],
        test_case_1["initial_cond"],
        test_case_1["galactic_model"],
    )

    assert np.isclose(
        derivatives_out,
        test_case_1["derivatives_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_dynamical_evolution(test_case_2):
    """
    Verifying (approximately) that the positions and velocities are correctly evolved in time.
    To do so, we use a simple finite differencing scheme, i.e., x_initial + x_derivative * time_step,
    to evaluate the first time step, only, and compare it to the output of solve_ivp for two object
    whose ages correspond to the first evaluated time step. With the above choices, the first time_step
    has a length of 10^4 years.
    """
    final_population_out, dyn_evol_dict_out = dyn.dynamical_evolution(
        test_case_2["initial_cond"],
        test_case_2["t_age"],
    )

    assert np.isclose(
        final_population_out,
        test_case_2["final_population_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()

    assert (
        dyn_evol_dict_out.keys()
        == test_case_2["dyn_evol_dict_expected"].keys()
    )

    for key in test_case_2["dyn_evol_dict_expected"].keys():
        assert (
            dyn_evol_dict_out[key].keys()
            == test_case_2["dyn_evol_dict_expected"][key].keys()
        )


def test_initialize_population_dyn(test_case_3, monkeypatch):
    """
    Verifying that the dictionary containing the dynamical properties at birth is properly initialized.
    """

    def mock_initial_population(*args, **kwargs):
        return MockInitialNeutronStarPopulation(*args, **kwargs)

    monkeypatch.setattr(
        ipop, "InitialNeutronStarPopulation", mock_initial_population
    )

    dict_out = dyn.initialize_population_dyn(test_case_3["age"])

    # Assertions.
    for key in dict_out:
        np.testing.assert_array_equal(
            dict_out[key], test_case_3["dict_expected"][key]
        )


def test_evolve_population_magrot(monkeypatch, test_case_4, tmp_path):
    """
    Check that the dictionary with the final dynamical properties is properly returned.
    """
    cfg["save_dyn_evolution"] = True  # Enable saving for testing file output.

    output_path = tmp_path

    def mock_dynamical_evolution(*args, **kwargs):
        return (
            test_case_4["mock_final_population"],
            test_case_4["dyn_evol_dict"],
        )

    monkeypatch.setattr(dyn, "dynamical_evolution", mock_dynamical_evolution)

    output_dict = dyn.evolve_population_dyn(
        test_case_4["dict_pop_initial_dyn"],
        output_path,
    )

    assert set(output_dict.keys()) == test_case_4["expected_keys"]

    # Check if the file was saved correctly.
    if cfg["save_dyn_evolution"]:
        magrot_file = output_path / "dyn_evolution.json"
        assert magrot_file.exists()


def test_check_angular_momentum_energy_conservation(test_case_5, monkeypatch):
    """
    Check that the info on the angular momentum and energy conservation is correctly logged.
    """
    monkeypatch.setattr(gm, "galactic_model", MockGalacticModel())

    # Mock the logger.
    log_calls = []

    class MockLogger:
        def info(self, message):
            log_calls.append(message)

    logger = MockLogger()

    # Run function.
    dyn.check_angular_momentum_energy_conservation(
        test_case_5["dict_initial"], test_case_5["dict_final"], logger
    )

    # Assertions.
    assert len(log_calls) == 2
    assert "Percentage variation of total energy of the system" in log_calls[0]
    assert (
        "Percentage variation of z-component of total angular momentum"
        in log_calls[1]
    )
