"""
Tests for the stellar_dynamics/dynamical_evolution module.

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

import pypopsyn.simulator.stellar_dynamics.dynamical_evolution as dyn
from pypopsyn.simulator.configuration import cfg

TOL = 1e-5

# Select the galactic model from Marchetti et al. (2019) for the test.
cfg["galactic_model"] = "gmM19"

# Set to save the time evolution output for testing purposes.
cfg["save_dyn_evolution"] = True


@pytest.fixture()
def test_case_1():

    import pypopsyn.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "initial_cond": np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "t": 0.0,
        "derivatives_expected": np.array(
            [0.0, 0.0, 0.0, -2.12522e-14, 0.0, -1.65820e-14]
        ),
        "galactic_model": gm.galactic_model,
    }

    return data


@pytest.fixture()
def test_case_2():

    import pypopsyn.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "initial_cond": np.array(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
        ),
        "t_age": np.array([1.0e4, 1.0e4]),
        "final_population_expected": np.array(
            [
                [1.0, 0.0, 1.0, -2.12522e-10, 0.0, -1.65820e-10],
                [1.0, 0.0, -1.0, -2.12522e-10, 0.0, 1.65820e-10],
            ]
        ),
        "dyn_evol_dict_expected": {
            0: {
                "t": None,
                "r(t)": None,
                "phi(t)": None,
                "z(t)": None,
                "v_r(t)": None,
                "v_phi(t)": None,
                "v_z(t)": None,
            },
            1: {
                "t": None,
                "r(t)": None,
                "phi(t)": None,
                "z(t)": None,
                "v_r(t)": None,
                "v_phi(t)": None,
                "v_z(t)": None,
            },
        },
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


def test_dynamical_evolution(test_case_2):
    """
    Verifying (approximately) that the positions and velocities are correctly evolved in time.
    To do so, we use a simple finite differencing scheme, i.e., x_initial + x_derivative * time_step,
    to evaluate the first time step, only, and compare it to the output of solve_ivp for two object
    whose ages correspond to the first evaluated time step. With the above choices, the first time_step
    has a length of 10^4 years.
    """
    final_population_out, dyn_evol_dict_out = dyn.dynamical_evolution(
        test_case_2["initial_cond"], test_case_2["t_age"],
    )
    assert np.isclose(
        final_population_out,
        test_case_2["final_population_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()

    assert (
        dyn_evol_dict_out.keys()
        == test_case_2["dyn_evol_dict_expected"].keys()
    )

    for key in test_case_2["dyn_evol_dict_expected"].keys():
        assert (
            dyn_evol_dict_out[key].keys()
            == test_case_2["dyn_evol_dict_expected"][key].keys()
        )
