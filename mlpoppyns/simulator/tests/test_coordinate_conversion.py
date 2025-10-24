"""
    Tests for the coordinate_conversion module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import mlpoppyns.simulator.stellar_dynamics.coordinate_conversions as coco

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


@pytest.fixture()
def test_case_4():
    data = {
        "dict_dyn": {
            "age": np.array([1e6, 2e6]),
            "r": np.array([8.0, 8.5]),
            "phi": np.array([0.1, 0.2]),
            "z": np.array([0.05, 0.1]),
            "v_r": np.array([100, 200]),
            "v_phi": np.array([220, 230]),
            "v_z": np.array([10, 20]),
            "idx": np.array([0, 1]),
        },
        "mock_polar_to_cartesian": (
            np.array([7.96, 8.46]),
            np.array([0.8, 1.7]),
        ),
        "mock_speed_cylindrical_to_cartesian": (
            np.array([110, 210]),
            np.array([210, 220]),
            np.array([15, 25]),
        ),
        "mock_galactocentric_to_icrs": (
            np.array([180.0, 181.0]),
            np.array([-30.0, -31.0]),
            np.array([8.0, 8.5]),
            np.array([5.0, 5.1]),
            np.array([-2.0, -2.1]),
            np.array([300, 310]),
        ),
        "mock_galactocentric_to_galactic": (
            np.array([50.0, 51.0]),
            np.array([10.0, 11.0]),
            np.array([8.0, 8.5]),
            np.array([1.0, 1.1]),
            np.array([-0.5, -0.6]),
            np.array([250, 260]),
        ),
        "dict_dyn_expected": {
            "age": np.array([1e6, 2e6]),
            "r": np.array([8.0, 8.5]),
            "phi": np.array([0.1, 0.2]),
            "z": np.array([0.05, 0.1]),
            "v_r": np.array([100, 200]),
            "v_phi": np.array([220, 230]),
            "v_z": np.array([10, 20]),
            "ra": np.array([180.0, 181.0]),
            "dec": np.array([-30.0, -31.0]),
            "l": np.array([50.0, 51.0]),
            "b": np.array([10.0, 11.0]),
            "dist": np.array([8.0, 8.5]),
            "pm_ra": np.array([5.0, 5.1]),
            "pm_dec": np.array([-2.0, -2.1]),
            "v_ls": np.array([300.0, 310.0]),
            "idx": np.array([0, 1]),
        },
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


def test_convert_cylindrical_to_all_coordinates(test_case_4, monkeypatch):
    """
    Verifying that the transformation of position and velocity components from cylindrical to
    ICRS and galactic coordinates is correct.
    """

    def mock_polar_to_cartesian(*args, **kwargs):
        return test_case_4["mock_polar_to_cartesian"]

    monkeypatch.setattr(coco, "polar_to_cartesian", mock_polar_to_cartesian)

    def mock_speed_cylindrical_to_cartesian(*args, **kwargs):
        return test_case_4["mock_speed_cylindrical_to_cartesian"]

    monkeypatch.setattr(
        coco,
        "speed_cylindrical_to_cartesian",
        mock_speed_cylindrical_to_cartesian,
    )

    def mock_galactocentric_to_icrs(*args, **kwargs):
        return test_case_4["mock_galactocentric_to_icrs"]

    monkeypatch.setattr(
        coco, "galactocentric_to_icrs", mock_galactocentric_to_icrs
    )

    def mock_galactocentric_to_galactic(*args, **kwargs):
        return test_case_4["mock_galactocentric_to_galactic"]

    monkeypatch.setattr(
        coco, "galactocentric_to_galactic", mock_galactocentric_to_galactic
    )

    dict_dyn_out = coco.convert_cylindrical_to_all_coordinates(
        test_case_4["dict_dyn"]
    )
    # Check that both dictionaries have the same keys
    assert dict_dyn_out.keys() == test_case_4["dict_dyn_expected"].keys()

    # Compare values for each key
    for key in test_case_4["dict_dyn_expected"]:
        assert np.array_equal(
            dict_dyn_out[key], test_case_4["dict_dyn_expected"][key]
        )
