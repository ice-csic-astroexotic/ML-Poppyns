"""
Tests for the initial_population module

    Authors:

        Vanessa Graber (graber @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""

import numpy as np
import pytest

import pypopsyn.simulator.initial_population as ipop
from pypopsyn.simulator.configuration import cfg

TOL = 1e-5

# Reduce the number of objects produced so that the computation time
# remains tractable for the tests.
cfg["NS_number"] = 5


@pytest.fixture()
def test_case_1():

    NS_population_initial = ipop.InitialNeutronStarPopulation()
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "x": position[2][selection],
        "y": position[3][selection],
        "z": position[4][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
    }

    return data


@pytest.fixture()
def test_case_2():

    NS_population_initial = ipop.InitialNeutronStarPopulation()
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "x": position[2][selection],
        "y": position[3][selection],
        "z": position[4][selection],
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
        test_case_1["x"], test_case_2["x"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_1["y"], test_case_2["y"], rtol=TOL, atol=1.0e-30
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
