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
        "pdf_vp_expected": 0.00119,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "r": 1.0,
        "z": 1.0,
        "v_virial_expected": 2.19947e-7,
    }

    return data


def test_pdf_proper_velocity(test_case_1):
    """
    Verifying that proper velocity distribution is correctly calculated.
    """
    pdf_vp_out = iv.pdf_proper_velocity(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vp_expected"] - pdf_vp_out) < TOL


def test_virial_orbital_velocity(test_case_2):
    """
    Verifying that the orbital virial velocity is evaluated correctly.
    """
    v_virial_out = iv.virial_orbital_velocity(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        v_virial_out, test_case_2["v_virial_expected"], rtol=TOL, atol=1.0e-30
    )
