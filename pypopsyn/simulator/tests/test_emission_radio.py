"""
Tests for the radio emission.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

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

import pypopsyn.simulator.multiband_emission.emission_radio as er
from pypopsyn.simulator.configuration import cfg

TOL = 1e-5

# Set the values of the configuration file for testing purposes.
cfg["NS_number"] = 2
cfg["L_radio_log10_mean"] = 26.0
cfg["L_radio_log10_sigma"] = 0.9
cfg["epsilon1"] = -1.5
cfg["epsilon2"] = 0.5


@pytest.fixture()
def test_case_1():
    data = {
        "r_em": cfg["r_em"],
        "P": np.array([0.1, 1.0]),
        "P_dot": np.array([1.0e-15]),
        "chi": np.array([np.pi / 3.0, np.pi / 4.0]),
        "theta_b": np.array([0.3, 0.1]),
        "los": np.array([0.3, 0.8]),
        "beam_aperture_expected": np.array([0.37612, 0.11894]),
        "w_expected": np.array([0.277910]),
        "beam_fraction_expected": np.array([0.51186, 0.14119]),
        "intercepted_expected": np.array([False, True]),
        "log10_L_0": np.array([26.0]),
        "L_radio_expected": np.array([3.162278e18]),
        "d": np.array([10.0]),
        "S_radio_expected": np.array([8.27960e-29]),
    }

    return data


def test_beam_aperture(test_case_1):
    """
    Verifying that for a given choice of spin period and emission radius the
    angular beam aperture is correctly calculated.
    """

    beam_aperture_out = er.beam_aperture(
        test_case_1["P"], test_case_1["r_em"],
    )

    assert np.isclose(
        test_case_1["beam_aperture_expected"],
        beam_aperture_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_pulse_width(test_case_1):
    """
    Verifying that for a given choice of the beam geometry and inclination angle,
    the pulse width is evaluated correctly.
    """
    chi = np.array(test_case_1["chi"][1])
    theta_b = np.array(test_case_1["theta_b"][1])
    los = np.array(test_case_1["los"][1])

    w_out = er.pulse_width(chi, theta_b, los)

    assert np.isclose(
        test_case_1["w_expected"], w_out, rtol=TOL, atol=1.0e-5
    ).all()


def test_beam_fraction(test_case_1):
    """
    Verifying that for a given choice of inclination angle and beam aperture the
    angular beam fraction is correctly calculated.
    """

    beam_fraction_out = er.beam_fraction(
        test_case_1["chi"], test_case_1["theta_b"],
    )

    assert np.isclose(
        test_case_1["beam_fraction_expected"],
        beam_fraction_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_los_intercept(test_case_1):
    """
    Verifying if the condition for the interception of the line of sight with the radio beam
    is correctly established.
    """

    intercepted_out = er.los_intercept(
        test_case_1["chi"], test_case_1["theta_b"], test_case_1["los"]
    )

    assert np.isclose(
        test_case_1["intercepted_expected"],
        intercepted_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_pdf_radio_luminosity(monkeypatch, test_case_1):
    """
    Verifying that the effective pulse width is computed correctly.
    """

    # Mocking the normalization of the luminosity distribution that is otherwise randomly determined.
    def mock_log10_L_0(*args, **kwargs):
        return test_case_1["log10_L_0"]

    monkeypatch.setattr(np.random, "normal", mock_log10_L_0)

    L_radio_out = er.pdf_radio_luminosity(
        np.array([test_case_1["P"][1]]), test_case_1["P_dot"],
    )

    assert np.isclose(
        test_case_1["L_radio_expected"], L_radio_out, rtol=TOL, atol=1.0e-7
    ).all()


def test_erg_flux_radio(test_case_1):
    """
    Verifying that the radio flux of a pulsar is correctly evaluated.
    """

    beam_frac = np.array(test_case_1["beam_fraction_expected"][1])

    S_radio_out = er.erg_flux_radio(
        test_case_1["L_radio_expected"],
        test_case_1["d"],
        beam_frac,
        test_case_1["w_expected"],
    )

    assert np.isclose(
        test_case_1["S_radio_expected"], S_radio_out, rtol=TOL, atol=1.0e-35
    ).all()
