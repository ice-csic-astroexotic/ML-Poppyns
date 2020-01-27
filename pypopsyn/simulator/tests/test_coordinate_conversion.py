import numpy as np
import pytest

import pypopsyn.simulator.coordinate_conversions as coco

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.5,
        "theta": 2.0,
        "x_expected": -0.62422,
        "y_expected": 1.36395,
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
        test_case_1["r"], test_case_1["theta"]
    )
    assert np.abs(test_case_1["x_expected"] - x_out) < TOL
    assert np.abs(test_case_1["y_expected"] - y_out) < TOL
