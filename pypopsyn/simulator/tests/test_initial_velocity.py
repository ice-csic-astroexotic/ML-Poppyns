"""
Test for the initial_velocity module

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""


import numpy as np
import pytest

import pypopsyn.simulator.initial_velocity as iv

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "v": 300,
        "pdf_vk_exp_expected": 0.00119,
        "pdf_vk_maxwell_expected": 0.00255,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "r": 1.0,
        "z": 1.0,
        "v_circular_expected": 2.19947e-7,
    }

    return data


def test_pdf_kick_velocity_exp(test_case_1):
    """
    Verifying that proper velocity distribution is correctly calculated for the exponential function.
    """
    pdf_vk_out = iv.pdf_kick_velocity_exp(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_exp_expected"] - pdf_vk_out) < TOL


def test_pdf_kick_velocity_maxwell(test_case_1):
    """
    Verifying that proper velocity distribution is correctly calculated for the Maxwell distribution.
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
