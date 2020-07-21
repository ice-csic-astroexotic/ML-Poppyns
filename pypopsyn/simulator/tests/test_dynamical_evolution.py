"""
Test for the dynamical_evolution module

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""


import numpy as np
import pytest

import pypopsyn.simulator.dynamical_evolution as dyn

TOL = 1e-5


@pytest.fixture()
def test_case_1():

    import pypopsyn.simulator.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "initial_cond": np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "t": np.linspace(0.0, 1.0, 10),
        "derivatives_expected": np.array(
            [0.0, 0.0, 0.0, -2.12522e-14, 0.0, -1.65820e-14]
        ),
        "galactic_model": gm.galactic_model,
    }

    return data


def test_dynamical_eq_system(test_case_1):
    """
    Verifying that the dynamical equation system evaluates the derivatives correctly.
    """
    derivatives_out = dyn.dynamical_eq_system(
        test_case_1["initial_cond"],
        test_case_1["t"],
        test_case_1["galactic_model"],
    )

    assert np.isclose(
        derivatives_out,
        test_case_1["derivatives_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
