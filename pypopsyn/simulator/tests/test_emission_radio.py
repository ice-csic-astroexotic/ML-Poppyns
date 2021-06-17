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


cfg["NS_number"] = 2


@pytest.fixture()
def test_case_1():
    data = {
        "r_em": cfg["r_em"],
        "P": np.array([0.1, 1.0]),
        "chi": np.array([np.pi / 3.0, np.pi / 4.0]),
        "theta_b": np.array([0.3, 0.1]),
        "los": np.array([0.3, 0.8]),
        "beam_aperture_expected": np.array([0.37612, 0.11894]),
        "w_expected": np.array([0.277910]),
        "beam_fraction_expected": np.array([0.51186, 0.14119]),
        "intercepted_expected": np.array([False, True]),
        "L_radio": np.array([5.0e19]),
        "d": np.array([10.0]),
        "S_radio_expected": np.array([1.30912e-27]),
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


def test_erg_flux_radio(test_case_1):
    """
    Verifying that the radio flux of a pulsar is correctly evaluated.
    """

    beam_frac = np.array(test_case_1["beam_fraction_expected"][1])

    S_radio_out = er.erg_flux_radio(
        test_case_1["L_radio"],
        test_case_1["d"],
        beam_frac,
        test_case_1["w_expected"],
    )

    assert np.isclose(
        test_case_1["S_radio_expected"], S_radio_out, rtol=TOL, atol=1.0e-35
    ).all()
