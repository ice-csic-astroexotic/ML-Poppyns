"""
    Tests for the initial_velocity module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import pypopsyn.simulator.stellar_dynamics.initial_velocity as iv
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5

# Set the galactic model from Marchetti et al. (2019) for the test.
cfg["galactic_model"] = "gmM19"
# Set the characteristic kick velocity of the exponential model for the test.
cfg["vk_c"] = 180.0
# Set the kick velocity dispersion of the Maxwell model for the test.
cfg["sigma_k"] = 265.0
# Set the parameters of the double Maxwell model for the test.
cfg["sigma_k_1"]: float = 55.0
cfg["sigma_k_2"]: float = 334.0
cfg["kick_weight"]: float = 0.19


@pytest.fixture()
def test_case_1():

    import pypopsyn.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "v": 300,
        "pdf_vk_exp_expected": 0.001749,
        "pdf_vk_maxwell_expected": 0.002033,
        "pdf_vk_2maxwell_expected": 0.001043,
    }

    return data


@pytest.fixture()
def test_case_2():

    import pypopsyn.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {"r": 1.0, "z": 1.0, "v_circular_expected": 1.12376e-7}

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


def test_pdf_kick_velocity_2maxwell(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the double Maxwell distribution.
    """
    pdf_vk_out = iv.pdf_kick_velocity_2maxwell(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_2maxwell_expected"] - pdf_vk_out) < TOL

    # Test if the error is properly raised when the weight is below 0.
    cfg["kick_weight"] = -0.1
    with pytest.raises(
        ValueError,
        match="The relative weight parameter of the km_2maxwell kick velocity model must be in "
        "the range 0 and 1.",
    ):
        iv.pdf_kick_velocity_2maxwell(test_case_1["v"])

    # Test if the error is properly raised when the weight is above 1.
    cfg["kick_weight"] = 1.1
    with pytest.raises(
        ValueError,
        match="The relative weight parameter of the km_2maxwell kick velocity model must be in "
        "the range 0 and 1.",
    ):
        iv.pdf_kick_velocity_2maxwell(test_case_1["v"])


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
