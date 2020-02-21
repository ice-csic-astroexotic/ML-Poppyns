import numpy as np
import pytest

import pypopsyn.simulator.dynamical_evolution as dyn

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "initial_cond": np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "t": np.linspace(0.0, 1.0, 10),
        "derivatives_expected": np.array(
            [0.0, 0.0, 0.0, -4.83765e-14, 0.0, -7.77990e-15]
        ),
    }

    return data


def test_dynamical_eq_system(test_case_1):
    """
    Verifying that the dynamical equation system evaluates the derivatives correctly
    """
    derivatives_out = dyn.dynamical_eq_system(
        test_case_1["initial_cond"], test_case_1["t"]
    )

    assert (
        np.abs(test_case_1["derivatives_expected"][0] - derivatives_out[0])
        < TOL
    )
    assert (
        np.abs(test_case_1["derivatives_expected"][1] - derivatives_out[1])
        < TOL
    )
    assert (
        np.abs(test_case_1["derivatives_expected"][2] - derivatives_out[2])
        < TOL
    )
    assert (
        np.abs(
            test_case_1["derivatives_expected"][3] * 1.0e14
            - derivatives_out[3] * 1.0e14
        )
        < TOL
    )
    assert (
        np.abs(test_case_1["derivatives_expected"][4] - derivatives_out[4])
        < TOL
    )
    assert (
        np.abs(
            test_case_1["derivatives_expected"][5] * 1.0e15
            - derivatives_out[5] * 1.0e15
        )
        < TOL
    )
