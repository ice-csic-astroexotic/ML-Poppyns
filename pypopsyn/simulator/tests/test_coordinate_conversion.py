"""
Tests for the stellar_dynamics/coordinate_conversion module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

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

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco

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


@pytest.fixture()
def test_case_3():
    data = {
        "v_r": 1.0,
        "v_phi": 1.0,
        "v_z": 1.0,
        "phi": np.pi / 4.0,
        "v_x_expected": 0.0,
        "v_y_expected": np.sqrt(2),
        "v_z_expected": 1.0,
    }

    return data


def test_check_radial_coordinate():
    """
    Verifying that a ValueError is raised if the radial coordinate is negative.
    """
    r = -0.1
    with pytest.raises(ValueError, match="Radial coordinate is out of range."):
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
    assert np.abs(test_case_3["v_x_expected"] - v_x_out) < TOL
    assert np.abs(test_case_3["v_y_expected"] - v_y_out) < TOL
    assert np.abs(test_case_3["v_z_expected"] - v_z_out) < TOL
