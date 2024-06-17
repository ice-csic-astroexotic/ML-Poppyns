"""
    Test for the period_derivative.py module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "B": np.array([1e12, 1e13, 1e14, 1e15]),
        "chi": np.array([0, np.pi / 2, np.pi, 2 * np.pi]),
        "P": np.array([1.0e-3, 1.0e-1, 1, 5]),
        "P_deriv_expected": np.array(
            [1.51007e-05, 3.020139e-05, 1.51007e-04, 3.020139e-03]
        ),
    }

    return data


def test_period_derivative(test_case_1):
    """
    Verifying that the period derivatives for a pulsar sample are evaluated correctly.
    """
    period_derivative_vect = np.vectorize(pdv.period_derivative)
    P_deriv_out = period_derivative_vect(
        test_case_1["B"], test_case_1["chi"], test_case_1["P"]
    )

    assert np.isclose(
        P_deriv_out,
        test_case_1["P_deriv_expected"],
        rtol=TOL,
        atol=1e-30,
    ).all()
