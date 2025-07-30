"""
    Tests for the initial_population_sam module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.initial_population as ipop
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5

# Reduce the number of objects produced so that the computation time
# remains tractable for the tests.
cfg["NS_number"] = 5
# Set a predefined seed for the tests.
cfg["seed_sampling"] = 42
# Select a spiral arm model to populate the Galaxy, not the electron density prescription.
cfg["sample_edm"]: bool = False
# For the tests, set the spiral arm pattern in the configuration file to
# the one from Faucher-Giguère & Kaspi (2006).
cfg["spiral_arms"] = "saFK06"
# Set the number of spiral arms to 4 for the tests.
cfg["arm_number"] = 4
# Select the Maxwell kick velocity model for the test.
cfg["kick_model"] = "km_maxwell"
# Select the Yusifov & Küçük (2004) radial density model for the test.
cfg["radial_model"] = "rmYK04"

# Initialize the spiral arm model.
sm.initialize_spiral_model()


@pytest.fixture()
def test_case_1():
    np.random.seed(cfg["seed_sampling"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    orb_velocity = NS_population_initial.orbital_velocity(
        position[0], position[2]
    )
    spin_period = NS_population_initial.period()
    magnetic_field = NS_population_initial.magnetic_field()
    chi = NS_population_initial.misalignment_angle()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
        "v_orb": orb_velocity[selection],
        "P": spin_period[selection],
        "B": magnetic_field[selection],
        "chi": chi[selection],
    }

    return data


@pytest.fixture()
def test_case_2():
    np.random.seed(cfg["seed_sampling"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    orb_velocity = NS_population_initial.orbital_velocity(
        position[0], position[2]
    )
    spin_period = NS_population_initial.period()
    magnetic_field = NS_population_initial.magnetic_field()
    chi = NS_population_initial.misalignment_angle()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
        "v_orb": orb_velocity[selection],
        "P": spin_period[selection],
        "B": magnetic_field[selection],
        "chi": chi[selection],
    }

    return data


@pytest.fixture()
def test_case_3():
    cfg["sample_edm"] = True
    np.random.seed(cfg["seed_sampling"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    orb_velocity = NS_population_initial.orbital_velocity(
        position[0], position[2]
    )
    spin_period = NS_population_initial.period()
    magnetic_field = NS_population_initial.magnetic_field()
    chi = NS_population_initial.misalignment_angle()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
        "v_orb": orb_velocity[selection],
        "P": spin_period[selection],
        "B": magnetic_field[selection],
        "chi": chi[selection],
    }

    return data


@pytest.fixture()
def test_case_4():
    cfg["sample_edm"] = True
    np.random.seed(cfg["seed_sampling"])
    NS_population_initial = ipop.InitialNeutronStarPopulation(cfg["NS_number"])
    age = NS_population_initial.age()
    position = NS_population_initial.position(t_age=age)
    kick_velocity = NS_population_initial.kick_velocity()
    orb_velocity = NS_population_initial.orbital_velocity(
        position[0], position[2]
    )
    spin_period = NS_population_initial.period()
    magnetic_field = NS_population_initial.magnetic_field()
    chi = NS_population_initial.misalignment_angle()
    selection = cfg["NS_number"] - 1

    data = {
        "age": age[selection],
        "r": position[0][selection],
        "phi": position[1][selection],
        "z": position[2][selection],
        "vk_r": kick_velocity[0][selection],
        "vk_phi": kick_velocity[1][selection],
        "vk_z": kick_velocity[2][selection],
        "v_orb": orb_velocity[selection],
        "P": spin_period[selection],
        "B": magnetic_field[selection],
        "chi": chi[selection],
    }

    return data


def test_age(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent ages.
    """
    assert np.isclose(
        test_case_1["age"], test_case_2["age"], rtol=TOL, atol=1.0e-30
    )


def test_position_spiral_arms(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent positions for the spiral arm model.
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


def test_position_edm(test_case_3, test_case_4):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent positions for the electron density model.
    """

    assert np.isclose(
        test_case_3["r"], test_case_4["r"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_3["phi"], test_case_4["phi"], rtol=TOL, atol=1.0e-30
    )
    assert np.isclose(
        test_case_3["z"], test_case_4["z"], rtol=TOL, atol=1.0e-30
    )


def test_kick_velocity(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent kick velocities.
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


def test_orbital_velocity(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent orbital velocities.
    """
    assert np.isclose(
        test_case_1["v_orb"], test_case_2["v_orb"], rtol=TOL, atol=1.0e-30
    )


def test_period(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent spin periods.
    """
    assert np.isclose(
        test_case_1["P"], test_case_2["P"], rtol=TOL, atol=1.0e-30
    )


def test_magnetic_field(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent spin periods.
    """
    assert np.isclose(
        test_case_1["B"], test_case_2["B"], rtol=TOL, atol=1.0e-30
    )


def test_misalignment(test_case_1, test_case_2):
    """
    Verifying that two separate instances of the InitialNeutronStarPopulation class
    have equivalent misalignment angles.
    """
    assert np.isclose(
        test_case_1["chi"], test_case_2["chi"], rtol=TOL, atol=1.0e-30
    )
