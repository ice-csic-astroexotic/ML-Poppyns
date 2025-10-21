"""
    Test for the period_derivative module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import mlpoppyns.simulator.basics.constants as const
import mlpoppyns.simulator.magneto_rotational_physics.period_derivative as pdv
from mlpoppyns.simulator.config_simulator import cfg

# Set the neutron parameters for testing purposes.
cfg["NS_mass"] = 1.4 * const.M_SUN
cfg["NS_radius"] = 1.1e6

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "B": 1e12,
        "chi": 0,
        "P": 1.0e-3,
        "P_deriv_expected": 1.51007e-05,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "B": np.array([1e12, 1e13, 1e14, 1e15]),
        "chi": np.array([0, np.pi / 2, np.pi, 2 * np.pi]),
        "P": np.array([1.0e-3, 1.0e-1, 1, 5]),
        "P_deriv_expected": np.array(
            [4.78839926e-13, 9.57679851e-13, 4.78839926e-12, 9.57679851e-11]
        ),
    }

    return data


def test_period_derivative(test_case_1):
    """
    Verifying that the period derivative for a pulsar is evaluated correctly.
    """

    P_deriv_out = pdv.period_derivative(
        test_case_1["B"],
        test_case_1["chi"],
        test_case_1["P"],
        cfg["NS_mass"],
        cfg["NS_radius"],
    )

    assert np.isclose(
        P_deriv_out,
        test_case_1["P_deriv_expected"],
        rtol=TOL,
        atol=1e-30,
    ).all()


def test_period_derivative_numpy(test_case_2):
    """
    Verifying that the period derivatives for a pulsar sample are evaluated correctly.
    """
    P_deriv_out = pdv.period_derivative_numpy(
        test_case_2["B"],
        test_case_2["chi"],
        test_case_2["P"],
        cfg["NS_mass"],
        cfg["NS_radius"],
    )

    assert np.isclose(
        P_deriv_out,
        test_case_2["P_deriv_expected"],
        rtol=TOL,
        atol=1e-30,
    ).all()
