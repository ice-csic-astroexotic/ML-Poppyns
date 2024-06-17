"""
    Tests for the coordinate_conversion.py module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": np.array([1.5, 10]),
        "phi": np.array([2.0, 1.5]),
        "x_expected": np.array([-0.62422, 0.70737]),
        "y_expected": np.array([1.36395, 9.97495]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "r": np.array([1.5, 10]),
        "theta": np.array([2.0, 1.5]),
        "psi": np.array([3.0, 1.0]),
        "x_expected": np.array([-1.35029, 5.38949]),
        "y_expected": np.array([0.19248, 8.39363]),
        "z_expected": np.array([-0.62422, 0.70737]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "v_r": np.array([1.0, 10.0]),
        "v_phi": np.array([1.0, 10.0]),
        "v_z": np.array([1.0, 10.0]),
        "phi": np.array([np.pi / 4.0, 1.0]),
        "v_x_expected": np.array([0.0, -3.011686789]),
        "v_y_expected": np.array([np.sqrt(2), 13.81773]),
        "v_z_expected": np.array([1.0, 10.0]),
    }

    return data


def test_check_radial_coordinate():
    """
    Verifying that a ValueError is raised if the radial coordinate is negative.
    """
    r = np.array([-0.1, 5])
    with pytest.raises(
        ValueError, match="One of the radial coordinates is out of range."
    ):
        coco.check_radial_coordinate(r)


def test_polar_to_cartesian(test_case_1):
    """
    Verifying that the conversion from polar to Cartesian coordinates is correct.
    """
    x_out, y_out = coco.polar_to_cartesian(
        test_case_1["r"], test_case_1["phi"]
    )
    assert np.isclose(
        test_case_1["x_expected"], x_out, rtol=TOL, atol=1.0e-30
    ).all()
    assert np.isclose(
        test_case_1["y_expected"], y_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_spherical_to_cartesian(test_case_2):
    """
    Verifying that the conversion from spherical to Cartesian coordinates is correct.
    """
    x_out, y_out, z_out = coco.spherical_to_cartesian(
        test_case_2["r"], test_case_2["theta"], test_case_2["psi"]
    )
    assert np.isclose(
        test_case_2["x_expected"], x_out, rtol=TOL, atol=1.0e-30
    ).all()
    assert np.isclose(
        test_case_2["y_expected"], y_out, rtol=TOL, atol=1.0e-30
    ).all()
    assert np.isclose(
        test_case_2["z_expected"], z_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_speed_cylindrical_to_cartesian(test_case_3):
    """
    Verifying that the transformation of velocity components from cylindrical to
    Cartesian galactocentric coordinates is correct.
    """
    v_x_out, v_y_out, v_z_out = coco.speed_cylindrical_to_cartesian(
        test_case_3["v_r"],
        test_case_3["v_phi"],
        test_case_3["v_z"],
        test_case_3["phi"],
    )
    assert np.isclose(
        test_case_3["v_x_expected"], v_x_out, rtol=TOL, atol=1.0e-10
    ).all()
    assert np.isclose(
        test_case_3["v_y_expected"], v_y_out, rtol=TOL, atol=1.0e-10
    ).all()
    assert np.isclose(
        test_case_3["v_z_expected"], v_z_out, rtol=TOL, atol=1.0e-10
    ).all()
