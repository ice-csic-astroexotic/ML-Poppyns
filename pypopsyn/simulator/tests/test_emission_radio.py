"""
    Tests for the emission_radio module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
            Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.multiband_emission.emission_radio as er
import utilities.samplers.random_sampler as rs
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5

# Set the values of the configuration file for testing purposes.
cfg["NS_number"] = 2
cfg["L_radio_log10_mean"] = 26.0
cfg["L_radio_log10_sigma"] = 0.9
cfg["epsilon_L"] = 0.5


@pytest.fixture()
def test_case_1():
    data = {
        "r_em": cfg["r_em"],
        "P": np.array([0.1, 1.0]),
        "P_dot": np.array([1.0e-15, 1.0e-13]),
        "chi": np.array([np.pi / 3.0, np.pi / 4.0]),
        "rho_b": np.array([0.3, 0.1]),
        "los": np.array([0.3, 0.8]),
        "solid_angle_expected": np.array([0.56126, 0.062780]),
        "beam_aperture_expected": np.array([0.37612, 0.11894]),
        "w_expected": np.array([0.277910]),
        "beam_fraction_expected": np.array([0.51186, 0.14119]),
        "intercepted_expected": np.array([False, True]),
        "log10_L_0": np.array([26.0, 25.0]),
        "L_radio_expected": np.array([1.0e20, 3.1622777e18]),
        "d": np.array([10.0, 5.0]),
        "f_survey": 1.4e9,
        "S_radio_expected": np.array([1.871269e-25, 2.116123e-25]),
        "S_radio_f_expected": np.array([4.151595e-13, 4.694829e-13]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "dataset_dict": {
            "P": np.array([6.28, 0.54]),
            "P_dot": np.array([1.0e-15, 1.0e-13]),
            "age": np.array([8.2e6, 1.7e6]),
            "l_gal": np.array([-2.05, -12.2]),
            "b_gal": np.array([-8.15, 7.51]),
            "dist": np.array([12.0, 6.59]),
            "B": np.array([2.32e11, 2.57e11]),
            "chi": np.array([0.99, 0.99]),
            "idx_det": np.array([1804, 2874]),
            "intercept_los_expected": np.array([False, True]),
            "los_rand": np.array([1.04, 1.05]),
            "l_radio_bol": np.array([7.85e24, 2.93e27]),
        },
        "dict_expected": {
            "age_det": np.array([1700000.0]),
            "l_det": np.array([-12.2]),
            "b_det": np.array([7.51]),
            "B_det": np.array([2.57e11]),
            "chi_det": np.array([0.99]),
            "P_det": np.array([0.54]),
            "P_dot_det": np.array([9.95040457e-17]),
            "w_int_s": np.array([0.03034442]),
            "w_int_s_full": np.array([0, 0.03034442]),
            "L_radio_bol": np.array([7.85e24, 2.93e27]),
            "S_radio_bol": np.array([1.15583818e-19, 4.31414761e-17]),
            "DM": np.array([130.08355713]),
            "idx_det": np.array([2874]),
            "intercepted_radio": np.array([False, True]),
        },
    }

    return data


def test_beam_aperture(test_case_1):
    """
    Verifying that for a given choice of spin period and emission radius the
    angular beam aperture is correctly calculated.
    """

    beam_aperture_out = er.beam_aperture(
        test_case_1["P"],
        test_case_1["r_em"],
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
    rho_b = np.array(test_case_1["rho_b"][1])
    los = np.array(test_case_1["los"][1])

    w_out = er.pulse_width(chi, rho_b, los)

    assert np.isclose(
        test_case_1["w_expected"], w_out, rtol=TOL, atol=1.0e-5
    ).all()


def test_solid_angle_radio_beams(test_case_1):
    """
    Verifying that for a given choice of beam aperture the
    solid angle covered by the radio beams is correctly calculated.
    """

    solid_angle_out = er.solid_angle_radio_beams(
        test_case_1["rho_b"],
    )

    assert np.isclose(
        test_case_1["solid_angle_expected"],
        solid_angle_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_beam_fraction(test_case_1):
    """
    Verifying that for a given choice of inclination angle and beam aperture the
    angular beam fraction is correctly calculated.
    """

    beam_fraction_out = er.beam_fraction(
        test_case_1["chi"],
        test_case_1["rho_b"],
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
        test_case_1["chi"], test_case_1["rho_b"], test_case_1["los"]
    )

    assert np.isclose(
        test_case_1["intercepted_expected"],
        intercepted_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_pdf_luminosity_radio(monkeypatch, test_case_1):
    """
    Verifying that the effective pulse width is computed correctly.
    """

    # Mocking the normalization of the luminosity distribution that is otherwise randomly determined.
    def mock_log10_L_0(*args, **kwargs):
        return test_case_1["log10_L_0"]

    monkeypatch.setattr(np.random, "normal", mock_log10_L_0)

    L_radio_out = er.pdf_luminosity_radio(
        np.array([test_case_1["P"]]),
        test_case_1["P_dot"],
    )

    assert np.isclose(
        test_case_1["L_radio_expected"], L_radio_out, rtol=TOL, atol=1.0e-7
    ).all()


def test_flux_radio(test_case_1):
    """
    Verifying that the radio flux of a pulsar is correctly evaluated.
    """

    S_radio_out = er.flux_radio(
        test_case_1["L_radio_expected"],
        test_case_1["d"],
        test_case_1["solid_angle_expected"],
    )

    assert np.isclose(
        test_case_1["S_radio_expected"], S_radio_out, rtol=TOL, atol=1.0e-35
    ).all()


def test_flux_density_radio(test_case_1):
    """
    Verifying that the radio flux density at a given frequency f is correctly evaluated.
    """

    S_radio_f_out = er.flux_density_radio(
        test_case_1["S_radio_expected"],
        test_case_1["f_survey"],
        spectral_index=-1.6,
        f_min=1.0e7,
        f_max=1.0e11,
    )

    assert np.isclose(
        test_case_1["S_radio_f_expected"],
        S_radio_f_out,
        rtol=TOL,
        atol=1.0e-35,
    ).all()


def test_calculate_radio_emission(monkeypatch, test_case_2):

    """
    Verifying that the radio emission is computed correctly.
    """

    def mock_los_rand(*args, **kwargs):
        return test_case_2["dataset_dict"]["los_rand"]

    monkeypatch.setattr(rs, "random_from_pdf", mock_los_rand)

    def mock_pdf_luminosity_radio(*args, **kwargs):
        return test_case_2["dataset_dict"]["l_radio_bol"]

    monkeypatch.setattr(er, "pdf_luminosity_radio", mock_pdf_luminosity_radio)

    emission_radio_dict_out = er.calculate_radio_emission(
        test_case_2["dataset_dict"]["P"],
        test_case_2["dataset_dict"]["age"],
        test_case_2["dataset_dict"]["l_gal"],
        test_case_2["dataset_dict"]["b_gal"],
        test_case_2["dataset_dict"]["dist"],
        test_case_2["dataset_dict"]["B"],
        test_case_2["dataset_dict"]["chi"],
        test_case_2["dataset_dict"]["idx_det"],
    )

    for key1 in emission_radio_dict_out.keys():

        assert np.isclose(
            emission_radio_dict_out[key1].all(),
            test_case_2["dict_expected"][key1].all(),
            rtol=TOL,
            atol=1.0e-30,
        )


def test_calculate_radio_emission_full(monkeypatch, test_case_2):

    """
    Verifying that the radio emission is computed correctly.
    """
    cfg["NS_number"] = 2

    def mock_los_rand(*args, **kwargs):
        return test_case_2["dataset_dict"]["los_rand"]

    monkeypatch.setattr(rs, "random_from_pdf", mock_los_rand)

    def mock_pdf_luminosity_radio(*args, **kwargs):
        return test_case_2["dataset_dict"]["l_radio_bol"]

    monkeypatch.setattr(er, "pdf_luminosity_radio", mock_pdf_luminosity_radio)

    (
        intercepted_radio,
        S_radio_bol,
        w_int_s,
        L_radio_bol,
    ) = er.calculate_radio_emission_full(
        test_case_2["dataset_dict"]["P"],
        test_case_2["dataset_dict"]["P_dot"],
        test_case_2["dataset_dict"]["dist"],
        test_case_2["dataset_dict"]["chi"],
    )
    assert np.isclose(
        test_case_2["dict_expected"]["intercepted_radio"],
        intercepted_radio,
        rtol=TOL,
        atol=1.0e-5,
    ).all()
    assert np.isclose(
        test_case_2["dict_expected"]["S_radio_bol"],
        S_radio_bol,
        rtol=TOL,
        atol=1.0e-5,
    ).all()
    assert np.isclose(
        test_case_2["dict_expected"]["w_int_s_full"],
        w_int_s,
        rtol=TOL,
        atol=1.0e-5,
    ).all()
    assert np.isclose(
        test_case_2["dict_expected"]["L_radio_bol"],
        L_radio_bol,
        rtol=TOL,
        atol=1.0e-5,
    ).all()
