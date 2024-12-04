"""
    Test for the initial_magnetic_field module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.magneto_rotational_physics.initial_magnetic_field as imf
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5

# Set the parameters of the distributions for testing purposes.
cfg["B_initial_log10_mean"]: float = 13.04
cfg["B_initial_log10_sigma"]: float = 0.53

cfg["B_initial_log10_mean1"]: float = 13.02
cfg["B_initial_log10_sigma1"]: float = 0.49
cfg["B_initial_log10_mean2"]: float = 14.5
cfg["B_initial_log10_sigma2"]: float = 0.5
cfg["B_initial_weight"]: float = 0.8

cfg["B_initial_log10_rise_mean"]: float = 13.02
cfg["B_initial_log10_rise_sigma"]: float = 0.49
cfg["B_initial_log10_decay_mean"]: float = 14.8
cfg["B_initial_log10_decay_sigma"]: float = 0.2
cfg["B_initial_log10_slope"]: float = -2.0


@pytest.fixture()
def test_case_1():

    data = {
        "log10B": np.array([13.0, 15.0]),
        "x": np.array([-0.1, 0.2]),
        "mean": 0.0,
        "sigma": 0.5,
        "norm": 1.0,
        "pdf_log10_magnetic_field_2normal_expected": np.array(
            [0.65256475, 0.09697372]
        ),
        "pdf_gaussian_custom_norm_expected": np.array(
            [0.98019867, 0.92311635]
        ),
        "pdf_log10_magnetic_field_smooth_tophat_expected": np.array(
            [2.77768526, 7.91447220e-4]
        ),
    }

    return data


def test_pdf_log10_magnetic_field_2normal(test_case_1):
    """
    Verifying that the double log-normal magnetic field distribution is properly calculated.
    """
    pdf_out = imf.pdf_log10_magnetic_field_2normal(test_case_1["log10B"])
    assert np.isclose(
        test_case_1["pdf_log10_magnetic_field_2normal_expected"],
        pdf_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_pdf_gaussian_custom_norm(test_case_1):
    """
    Verifying that the Gaussian with custom normalization is properly calculated.
    """
    pdf_out = imf.pdf_gaussian_custom_norm(
        test_case_1["x"],
        test_case_1["mean"],
        test_case_1["sigma"],
        test_case_1["norm"],
    )
    assert np.isclose(
        test_case_1["pdf_gaussian_custom_norm_expected"],
        pdf_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_pdf_log10_magnetic_field_smooth_tophat(test_case_1):
    """
    Verifying that the smooth top-hat distribution is properly calculated.
    """
    pdf_out = imf.pdf_log10_magnetic_field_smooth_tophat(test_case_1["log10B"])
    assert np.isclose(
        test_case_1["pdf_log10_magnetic_field_smooth_tophat_expected"],
        pdf_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_initial_magnetic_field_lognormal(test_case_1):
    """
    Check that the array of initial magnetic fields has the correct length, corresponding
    to the number of pulsars in our sample.
    """
    B_initial_out = imf.initial_magnetic_field_lognormal(
        cfg["B_initial_log10_mean"],
        cfg["B_initial_log10_sigma"],
        cfg["NS_number"],
    )
    assert len(B_initial_out) == cfg["NS_number"]


def test_initial_magnetic_field_double_lognormal(test_case_1):
    """
    Check that the array of initial magnetic fields has the correct length, corresponding
    to the number of pulsars in our sample.
    """
    B_initial_out = imf.initial_magnetic_field_double_lognormal(
        cfg["NS_number"]
    )
    assert len(B_initial_out) == cfg["NS_number"]


def test_initial_magnetic_field_smooth_tophat(test_case_1):
    """
    Check that the array of initial magnetic fields has the correct length, corresponding
    to the number of pulsars in our sample.
    """
    B_initial_out = imf.initial_magnetic_field_smooth_tophat(cfg["NS_number"])
    assert len(B_initial_out) == cfg["NS_number"]
