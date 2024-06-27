"""
    Tests for the initial_population_sam module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.initial_population_sam as ipop
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5

# Reduce the number of objects produced so that the computation time
# remains tractable for the tests.
cfg["NS_number"] = 5
# Set a predefined seed for the tests.
cfg["seed"] = 42

# For the tests, set the spiral arm pattern in the configuration file to
# the one from Faucher-Giguère & Kaspi (2006).
cfg["spiral_arms"] = "saFK06"
# Set the number of spiral arms to 4 for the tests.
cfg["arm_number"] = 4
# Select the Maxwell kick velocity model for the test.
cfg["kick_model"] = "km_maxwell"
# Select the Yusifov & Küçük (2004) radial density model for the test.
cfg["radial_model"] = "rmYK04"

sm.initialize_spiral_model()


@pytest.fixture()
def test_case_1():
    np.random.seed(cfg["seed"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(
        t_age=age, spiral_model=sm.spiral_model
    )
    kick_velocity = NS_population_initial.kick_velocity()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
    }

    return data


@pytest.fixture()
def test_case_2():
    np.random.seed(cfg["seed"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(
        t_age=age, spiral_model=sm.spiral_model
    )
    kick_velocity = NS_population_initial.kick_velocity()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
    }

    return data


@pytest.mark.skipif(cfg["seed"] is None, reason="seed is not specified")
def test_age(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent ages provided that a seed is set in the configuration file
    for random number selection.
    """
    assert np.isclose(
        test_case_1["age"], test_case_2["age"], rtol=TOL, atol=1.0e-30
    )


@pytest.mark.skipif(cfg["seed"] is None, reason="seed is not specified")
def test_position(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent positions provided that a seed is set in the configuration file
    for random number selection.
    """
    assert np.isclose(
        test_case_1["r"], test_case_2["r"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_1["phi"], test_case_2["phi"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_1["z"], test_case_2["z"], rtol=TOL, atol=1.0e-30
    )


@pytest.mark.skipif(cfg["seed"] is None, reason="seed is not specified")
def test_kick_velocity(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent kick velocities provided that a seed is set in the configuration
    file for random number selection.
    """
    assert np.isclose(
        test_case_1["vk_r"], test_case_2["vk_r"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_1["vk_phi"], test_case_2["vk_phi"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_1["vk_z"], test_case_2["vk_z"], rtol=TOL, atol=1.0e-30
    )
