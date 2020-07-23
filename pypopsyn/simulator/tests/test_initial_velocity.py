"""
Tests for the initial_velocity module.

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

import pypopsyn.simulator.initial_velocity as iv
from pypopsyn.simulator.configuration import cfg

TOL = 1e-5

# Select the galactic model from Marchetti et al. (2019) for the test.
cfg["galactic_model"] = "gmM19"


@pytest.fixture()
def test_case_1():

    import pypopsyn.simulator.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "v": 300,
        "pdf_vk_exp_expected": 0.00119,
        "pdf_vk_maxwell_expected": 0.002033,
    }

    return data


@pytest.fixture()
def test_case_2():

    import pypopsyn.simulator.galactic_model as gm

    gm.initialize_galactic_model()

    data = {"r": 1.0, "z": 1.0, "v_circular_expected": 1.45781e-7}

    return data


def test_pdf_kick_velocity_exp(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the exponential function.
    """
    pdf_vk_out = iv.pdf_kick_velocity_exp(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_exp_expected"] - pdf_vk_out) < TOL


def test_pdf_kick_velocity_maxwell(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the Maxwell distribution.
    """
    pdf_vk_out = iv.pdf_kick_velocity_maxwell(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_maxwell_expected"] - pdf_vk_out) < TOL


def test_circular_velocity(test_case_2):
    """
    Verifying that the circular velocity is evaluated correctly.
    """
    v_circular_out = iv.circular_velocity(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        v_circular_out,
        test_case_2["v_circular_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
