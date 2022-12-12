"""
Tests for the radio survey module.

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

import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.multiband_surveys.survey_radio as sr

TOL = 1e-5

PMPS_par_path = "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json"
PMPS = sr.SurveyRadio(PMPS_par_path)


@pytest.fixture()
def test_case_1():
    data = {
        "w_int": np.array([1.0e-3, 1.0e-4]),
        "DM": np.array([100, 1000]),
        "tau_sc": np.array([1.0e-4, 1.0e-2]),
        "f": 1.4e9,
        "channel_width": 3.0e6,
        "t_samp": 250e-6,
        "tau_DM_expected": np.array([0.00090717, 0.00907173]),
        "w_eff_expected": np.array([0.00137676, 0.0135044]),
        "l_gal": np.array([-90.0, 0.0]),
        "b_gal": np.array([0.0, 20.0]),
        "RA": np.array([30.0, 200.0]),
        "DEC": np.array([-30.0, 50.0]),
        "T_sky_expected": np.array([3.00651602, 1.25853108]),
        "coverage_expected": np.array([True, False], dtype=bool),
        "offset2": np.array([5, 10]),
        "G_expected": np.array([0.68485507, 0.63813124]),
        "S_radio_int": np.array([0.01, 100]),
        "S_radio_obs_expected": np.array([0.00726343, 0.74049939]),
        "S_radio_obs_mean_expected": np.array([9.99999989e-5, 0.999999996]),
        "P": np.array([0.1, 0.01]),
        "SNR_expected": np.array([1285.8829469, 0.0]),
        "detected_expected": np.array([True, False], dtype=bool),
    }

    return data


def test_smearing_in_channel(test_case_1):
    """
    Verifying that the inter-channel dispersion smearing is computed correctly.
    """

    tau_DM_out = sr.smearing_in_channel(
        test_case_1["DM"], test_case_1["channel_width"], test_case_1["f"]
    )

    assert np.isclose(
        test_case_1["tau_DM_expected"], tau_DM_out, rtol=TOL, atol=1.0e-7
    ).all()


def test_effective_pulse_width(monkeypatch, test_case_1):
    """
    Verifying that the effective pulse width is computed correctly.
    """

    # Mocking the scattering timescale that is otherwise randomly determined.
    def mock_tau_sc(*args, **kwargs):
        return test_case_1["tau_sc"]

    monkeypatch.setattr(edm, "compute_tau_sc", mock_tau_sc)

    w_eff_out = sr.effective_pulse_width(
        test_case_1["w_int"],
        test_case_1["DM"],
        test_case_1["channel_width"],
        test_case_1["f"],
        test_case_1["t_samp"],
    )

    assert np.isclose(
        test_case_1["w_eff_expected"], w_eff_out, rtol=TOL, atol=1.0e-7
    ).all()


def test_flux_radio_obs(test_case_1):
    """
    Verifying that the observed radio flux of a pulsar is correctly evaluated.
    """

    S_radio_obs_out = sr.flux_radio_obs(
        test_case_1["S_radio_int"],
        test_case_1["w_int"],
        test_case_1["w_eff_expected"],
    )

    assert np.isclose(
        test_case_1["S_radio_obs_expected"],
        S_radio_obs_out,
        rtol=TOL,
        atol=1.0e-35,
    ).all()


def test_flux_radio_obs_period_average(test_case_1):
    """
    Verifying that the observed radio period-averaged flux of a pulsar is correctly evaluated.
    """

    S_radio_obs_mean_out = sr.flux_radio_obs_period_average(
        test_case_1["S_radio_obs_expected"],
        test_case_1["P"],
        test_case_1["w_eff_expected"],
    )

    assert np.isclose(
        test_case_1["S_radio_obs_mean_expected"],
        S_radio_obs_mean_out,
        rtol=TOL,
        atol=1.0e-35,
    ).all()


def test_sky_temperature_approx(test_case_1):
    """
    Verifying that the sky temperature is computed correctly.
    """

    T_sky_out = sr.sky_temperature_approx(
        test_case_1["l_gal"], test_case_1["b_gal"], test_case_1["f"]
    )

    assert np.isclose(
        test_case_1["T_sky_expected"], T_sky_out, rtol=TOL, atol=1.0e-5
    ).all()


def test_sky_coverage_PMPS(test_case_1):
    """
    Verifying that the sky coverage of a survey is computed correctly.
    """

    coverage_out = PMPS.sky_coverage(
        test_case_1["RA"],
        test_case_1["DEC"],
        test_case_1["l_gal"],
        test_case_1["b_gal"],
    )

    assert test_case_1["coverage_expected"].all() == coverage_out.all()


def test_gain_gaussian_beam_PMPS(test_case_1):
    """
    Verifying that the Gaussian beam gain for an offset observation is computed correctly.
    """

    G_out = PMPS.gain_gaussian_beam(test_case_1["offset2"])

    assert np.isclose(
        test_case_1["G_expected"], G_out, rtol=TOL, atol=1.0e-5
    ).all()


def test_radiometer_equation_PMPS(test_case_1):
    """
    Verifying that the signal-to-noise values are computed correctly using the radiometer equation.
    """

    SNR_out = PMPS.radiometer_equation(
        test_case_1["S_radio_obs_expected"],
        test_case_1["G_expected"],
        test_case_1["w_eff_expected"],
        test_case_1["P"],
        test_case_1["T_sky_expected"],
    )

    assert np.isclose(
        test_case_1["SNR_expected"], SNR_out, rtol=TOL, atol=1.0e-5
    ).all()


def test_detect_PMPS(monkeypatch, test_case_1):
    """
    Verifying that the pulsars are correctly detected by the survey.
    """

    # Mocking the sky temperature.
    def mock_T_sky(*args, **kwargs):
        return test_case_1["T_sky_expected"]

    monkeypatch.setattr(sr, "sky_temperature_H81refined", mock_T_sky)

    detected_out = PMPS.detect(
        test_case_1["S_radio_obs_expected"],
        test_case_1["l_gal"],
        test_case_1["b_gal"],
        test_case_1["w_eff_expected"],
        test_case_1["P"],
    )

    assert test_case_1["detected_expected"].all() == detected_out.all()
