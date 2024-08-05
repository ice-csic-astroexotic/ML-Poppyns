"""
    Test for the misalignment_angle_derivative module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_derivative as madv

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "B": np.array([1e12, 1e13, 1e14, 1e15]),
        "chi": np.array([0, np.pi / 3, 1.4 * np.pi, 2.6 * np.pi]),
        "P": np.array([1.0e-2, 1.0e-1, 1.0, 5.0]),
        "chi_deriv_expected": np.array(
            [0.0, -6.538794e-05, -4.437982e-05, 1.775193e-04]
        ),
    }

    return data


def test_misalignment_angle_derivative(test_case_1):
    """
    Verifying that the misalignment angle derivatives for a pulsar sample are evaluated correctly.
    """
    misalignment_angle_derivative_vect = np.vectorize(
        madv.misalignment_angle_derivative
    )
    chi_deriv_out = misalignment_angle_derivative_vect(
        test_case_1["B"],
        test_case_1["chi"],
        test_case_1["P"],
    )

    assert np.isclose(
        chi_deriv_out,
        test_case_1["chi_deriv_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
