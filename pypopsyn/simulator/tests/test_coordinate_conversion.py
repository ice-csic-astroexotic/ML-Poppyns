"""
Test for the coordinate_conversion module

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""


import numpy as np
import pytest

import pypopsyn.simulator.coordinate_conversions as coco

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.5,
        "phi": 2.0,
        "x_expected": -0.62422,
        "y_expected": 1.36395,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "r": 1.5,
        "theta": 2.0,
        "psi": 3.0,
        "x_expected": -1.35029,
        "y_expected": 0.19248,
        "z_expected": -0.62422,
    }

    return data


def test_check_radial_coordinate():
    """
    Verifying that a ValueError is raised if the radial coordinate is negative.
    """
    r = -0.1
    with pytest.raises(ValueError, match="Radial coordinate is out of range"):
        coco.check_radial_coordinate(r)


def test_polar_to_cartesian(test_case_1):
    """
    Verifying that the conversion from polar to Cartesian coordinates is correct.
    """
    x_out, y_out = coco.polar_to_cartesian(
        test_case_1["r"], test_case_1["phi"]
    )
    assert np.abs(test_case_1["x_expected"] - x_out) < TOL
    assert np.abs(test_case_1["y_expected"] - y_out) < TOL


def test_spherical_to_cartesian(test_case_2):
    """
    Verifying that the conversion from spherical to Cartesian coordinates is correct.
    """
    x_out, y_out, z_out = coco.spherical_to_cartesian(
        test_case_2["r"], test_case_2["theta"], test_case_2["psi"]
    )
    assert np.abs(test_case_2["x_expected"] - x_out) < TOL
    assert np.abs(test_case_2["y_expected"] - y_out) < TOL
    assert np.abs(test_case_2["z_expected"] - z_out) < TOL
