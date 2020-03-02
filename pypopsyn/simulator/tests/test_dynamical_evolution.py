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

    assert np.isclose(
        derivatives_out,
        test_case_1["derivatives_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
