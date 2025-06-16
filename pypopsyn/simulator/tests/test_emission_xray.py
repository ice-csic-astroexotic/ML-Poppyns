"""
Tests for the X-ray emission module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

"""

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.multiband_emission.emission_xray as xem
from pypopsyn.simulator.config_simulator import cfg

# Set the neutron star radius for testing purposes.
cfg["NS_radius"] = 1.1e6

# Mock the gr_correction global variable to use the updated value of NS_radius for the test.
value = (
    1 - (2 * const.G * cfg["NS_mass"]) / (const.C**2 * cfg["NS_radius"])
) ** 0.5

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "Lx": np.array([3.0e35, 2.0e34, 2.0e33]),
        "T": np.array([1.0e6, 2.0e6]),
        "E": np.logspace(1.0, np.log10(20000), 4) * const.EV_TO_ERG,
        "E_0": np.logspace(1.0, np.log10(20000), 4) * const.EV_TO_ERG,
        "tau_0": np.array([1.0, 5.0, 10.0]),
        "beta_T": np.array([0.1, 0.3, 0.5]),
        "I_ph_source": np.array(
            [
                [3.0e26, 4.0e27, 2.0e28, 2.0e22],
                [1.0e26, 2.0e27, 1.0e28, 1.0e22],
                [5.0e26, 6.0e27, 3.0e28, 3.0e22],
            ]
        ),
        "n_reflections": 6,
        "B": np.array([1.0e13, 1.0e14, 1.0e15]),
        "RA": np.array([60.0, 120.0, 250.0]),
        "DEC": np.array([35.0, 45.0, 55.0]),
        "d": np.array([5.0, 10.0, 15.0]),
        "T_expected": np.array(
            [3835798.38544252, 1949094.77496194, 1096056.53867368]
        ),
        "I_bb_expected": np.array(
            [
                [2.55670678e26, 1.89805931e28, 1.25804071e24, 4.03385349e-66],
                [5.26614782e26, 5.84076633e28, 1.25833142e28, 1.00757951e-15],
            ]
        ),
        "n_plus_without_delta_expected": np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ),
        "n_minus_expected": np.array(
            [
                [
                    [5.03248129e10, 0.0, 0.0, 0.0],
                    [0.0, 3.99428305e09, 0.0, 0.0],
                    [0.0, 0.0, 3.17026456e08, 0.0],
                    [0.0, 0.0, 0.0, 2.51624065e07],
                ],
                [
                    [3.51144536e10, 0.0, 0.0, 0.0],
                    [0.0, 2.78703603e09, 0.0, 0.0],
                    [0.0, 0.0, 2.21207196e08, 0.0],
                    [0.0, 0.0, 0.0, 1.75572268e07],
                ],
                [
                    [
                        2.86392310e10,
                        1.84018096e08,
                        7.15342265e06,
                        5.28975712e05,
                    ],
                    [0.0, 2.27309727e09, 1.46055260e07, 5.67767532e05],
                    [0.0, 0.0, 1.80415850e08, 1.15924137e06],
                    [0.0, 0.0, 0.0, 1.43196155e07],
                ],
            ]
        ),
        "rcs_spectrum_expected": np.array(
            [
                [2.40932233e29, 4.88650350e30, 2.44325175e31, 1.79400851e22],
                [4.47753265e25, 9.76982358e26, 4.88491179e27, 1.03804395e21],
                [4.62415903e25, 2.07720449e26, 7.94215165e26, 2.44894218e20],
            ]
        ),
        "beta_T_expected": np.array([0.001, 0.3, 0.3]),
        "tau_res_expected": np.array([0.001, 1.0, 10.0]),
        "flux_bb_expected": np.array(
            [[6.35986131e-11, 1.05078767e-12, 3.53004896e-14]]
        ),
        "flux_rcs_expected": np.array(
            [6.37348633e-11, 1.07255307e-12, 5.85475838e-14]
        ),
        "N_H_expected": np.array(
            [2.38085937e21, 5.48339844e20, 2.89306641e20]
        ),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "age": np.array([1e4, 2e5, 1e7]),
        "ra": np.array([50.0, 250.0, 30.0]),
        "dec": np.array([-50.0, 50.0, 30.0]),
        "dist": np.array([2.0, 10.0, 5.0]),
        "B_initial": np.array([1e12, 1e14, 1e13]),
        "B": np.array([1e12, 1e14, 1e13]),
        "L_x_threshold": 1e29,
        "dummy_L_x_interpolator": RectBivariateSpline(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        ),
        "xray_bright_mask_expected": np.array([False, True, False]),
        "L_x_therm_expected": np.array([1e33]),
        "mock_L_x_therm": np.array([1e28, 1e33, 1e27]),
        "mock_S_x_rcs_abs": np.array([3.0e-12]),
        "mock_S_x_bb_abs": np.array([2.0e-12]),
        "mock_N_H": np.array([2.0e-21]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "age": np.array([50, 200, 800, 2000]),
        "B_initial": np.array([1e14, 5e12, 2e13, 9e12]),
        "outburst_mask_expected": np.array([True, False, False, False]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "dict_final_pop": {
            "age": np.array([1e6, 2e6]),
            "l": np.array([-50.0, 50.0]),
            "b": np.array([-20.0, 10.0]),
            "ra": np.array([50.0, 250.0]),
            "dec": np.array([-50.0, 50.0]),
            "dist": np.array([2.0, 10.0]),
            "pm_ra": np.array([-50.0, 50.0]),
            "pm_dec": np.array([-50.0, 50.0]),
            "v_ls": np.array([-50.0, 50.0]),
            "P": np.array([0.01, 0.5]),
            "P_dot": np.array([1.0e-11, 1.0e-12]),
            "B_initial": np.array([1e12, 1e14]),
            "B": np.array([1e12, 1e14]),
            "chi": np.array([1.0, 2.0]),
            "idx": np.array([0, 1]),
            "coverage_radio_PMPS": np.array([True, False]),
            "coverage_radio_HTRU_low": np.array([True, False]),
            "coverage_radio_HTRU_mid": np.array([True, False]),
            "coverage_radio": np.array([True, False]),
            "coverage_xray": np.array([True, True]),
        },
        "L_x_threshold": 1e29,
        "S_x_abs_threshold": 1e-15,
        "dummy_L_x_interpolator": RectBivariateSpline(
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        ),
        "xray_bright_mask": np.array([True, True]),
        "L_x_therm": np.array([1e33, 1e34]),
        "S_x_rcs_abs": np.array([3.0e-12, 4.0e-16]),
        "S_x_bb_abs": np.array([2.0e-12, 3.0e-16]),
        "N_H": np.array([2.0e-21, 3.0e-21]),
        "expected_keys": [
            "age",
            "ra",
            "dec",
            "l",
            "b",
            "N_H",
            "dist",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "B_initial",
            "B",
            "chi",
            "P",
            "P_dot",
            "L_x_therm",
            "S_x_rcs_abs",
            "S_x_bb_abs",
            "idx",
            "coverage_radio",
            "coverage_radio_HTRU_low",
            "coverage_radio_HTRU_mid",
            "coverage_radio_PMPS",
            "coverage_xray",
            "outburst",
        ],
    }

    return data


def test_T_from_Lx(test_case_1, monkeypatch):
    """
    Verifying that for a given X-ray luminosity the temperature is correctly calculated.
    """

    # Mock the gr_correction global variable to use the updated value of NS_radius for the test.
    monkeypatch.setattr(xem, "gr_correction", value)

    T_out = xem.T_from_Lx(
        test_case_1["Lx"],
    )

    assert np.isclose(
        test_case_1["T_expected"],
        T_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_blackbody_intensity_spectrum(test_case_1):
    """
    Verifying that for a given temperature the black-body intensity spectrum is correctly calculated.
    """

    I_bb_out = xem.blackbody_intensity_spectrum(
        test_case_1["E"], test_case_1["T"]
    )

    assert np.isclose(
        test_case_1["I_bb_expected"],
        I_bb_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_n_plus_without_delta(test_case_1):
    """
    Verifying that the transmission function n+ without the Dirac delta term is correctly calculated.
    """

    n_plus_without_delta_out = xem.n_plus_without_delta(
        test_case_1["E"],
        test_case_1["E_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["n_plus_without_delta_expected"],
        n_plus_without_delta_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_n_minus(test_case_1):
    """
    Verifying that the reflection function n- is correctly calculated.
    """

    n_minus_out = xem.n_minus(
        test_case_1["E"],
        test_case_1["E_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["n_minus_expected"],
        n_minus_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_resonant_cyclotron_scat_spectrum(test_case_1):
    """
    Verifying that the RCS spectrum is correctly calculated.
    """

    rcs_spectrum_out = xem.resonant_cyclotron_scat_spectrum(
        test_case_1["E"],
        test_case_1["E"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
        test_case_1["I_ph_source"],
        test_case_1["n_reflections"],
    )

    assert np.isclose(
        test_case_1["rcs_spectrum_expected"],
        rcs_spectrum_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_beta_plasma(test_case_1):
    """
    Verifying that the plasma velocity is correctly estimated.
    """

    beta_T_out = xem.beta_plasma(
        test_case_1["B"],
    )

    assert np.isclose(
        test_case_1["beta_T_expected"],
        beta_T_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_resonant_optical_depth(test_case_1):
    """
    Verifying that the resonant optical depth is correctly estimated.
    """

    tau_res_out = xem.resonant_optical_depth(
        test_case_1["B"],
    )

    assert np.isclose(
        test_case_1["tau_res_expected"],
        tau_res_out,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_flux_xray_absorbed(test_case_1, monkeypatch):
    """
    Verifying that absorbed X-ray flux is correctly estimated.
    """

    # Mock the gr_correction global variable to use the updated value of NS_radius for the test.
    monkeypatch.setattr(xem, "gr_correction", value)

    flux_bb_out, flux_rcs_out, N_H_out = xem.flux_xray_absorbed(
        test_case_1["Lx"],
        test_case_1["B"],
        test_case_1["RA"],
        test_case_1["DEC"],
        test_case_1["d"],
    )

    assert np.isclose(
        test_case_1["flux_bb_expected"],
        flux_bb_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()

    assert np.isclose(
        test_case_1["flux_rcs_expected"],
        flux_rcs_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()

    assert np.isclose(
        test_case_1["N_H_expected"],
        N_H_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()


def test_calculate_xray_emission(test_case_2, monkeypatch):
    """
    Verifying that absorbed X-ray flux is correctly estimated.
    """

    def mock_interpolator(*args, **kwargs):
        return test_case_2["mock_L_x_therm"]

    monkeypatch.setattr(RectBivariateSpline, "ev", mock_interpolator)

    def mock_flux_xray_absorbed(*args, **kwargs):
        return (
            test_case_2["mock_S_x_bb_abs"],
            test_case_2["mock_S_x_rcs_abs"],
            test_case_2["mock_N_H"],
        )

    monkeypatch.setattr(xem, "flux_xray_absorbed", mock_flux_xray_absorbed)

    (
        xray_bright_mask_out,
        L_x_therm_out,
        S_x_bb_abs_out,
        S_x_rcs_abs_out,
        N_H_out,
    ) = xem.calculate_xray_emission(
        test_case_2["B"],
        test_case_2["B_initial"],
        test_case_2["age"],
        test_case_2["ra"],
        test_case_2["dec"],
        test_case_2["dist"],
        test_case_2["dummy_L_x_interpolator"],
        test_case_2["L_x_threshold"],
    )

    assert np.all(
        xray_bright_mask_out == test_case_2["xray_bright_mask_expected"]
    )

    assert np.isclose(
        test_case_2["L_x_therm_expected"],
        L_x_therm_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()

    assert np.isclose(
        test_case_2["mock_S_x_bb_abs"],
        S_x_bb_abs_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()

    assert np.isclose(
        test_case_2["mock_S_x_rcs_abs"],
        S_x_rcs_abs_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()

    assert np.isclose(
        test_case_2["mock_N_H"],
        N_H_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()


def test_outburst_filter(test_case_3, monkeypatch):

    # Fixed uniform return values based on age group logic.
    def mock_uniform(low, high, size):
        if low == 0.40 and high == 0.85:
            return np.full(size, 0.7)
        elif low == 0.1 and high == 0.4:
            return np.full(size, 0.2)
        elif low == 0.05 and high == 0.2:
            return np.full(size, 0.1)
        elif low == 0.0 and high == 0.05:
            return np.full(size, 0.03)
        else:
            raise ValueError("Unexpected uniform call")

    # Fixed rand values: these simulate the draw to compare against probability.
    def mock_rand(size):
        return np.array([0.6, 0.3, 0.4, 0.02])

    monkeypatch.setattr(np.random, "uniform", mock_uniform)
    monkeypatch.setattr(np.random, "rand", mock_rand)

    outburst_mask_out = xem.outburst_filter(
        test_case_3["B_initial"], test_case_3["age"]
    )

    assert np.all(outburst_mask_out == test_case_3["outburst_mask_expected"])


def test_xray_population(test_case_4, monkeypatch):
    """
    Check that the dictionary with the properties of the neutron stars that are detected in X-rays is properly returned.
    """

    def mock_calculate_xray_emission(*args, **kwargs):
        return (
            test_case_4["xray_bright_mask"],
            test_case_4["L_x_therm"],
            test_case_4["S_x_bb_abs"],
            test_case_4["S_x_rcs_abs"],
            test_case_4["N_H"],
        )

    monkeypatch.setattr(
        xem, "calculate_xray_emission", mock_calculate_xray_emission
    )

    out_dict = xem.xray_population(
        test_case_4["dict_final_pop"],
        test_case_4["dummy_L_x_interpolator"],
        test_case_4["L_x_threshold"],
    )
    # Verify that the keys are correct.
    assert set(out_dict.keys()) == set(test_case_4["expected_keys"])
    # Verify that the output dictionary contains at most the same number of stars as the input one.
    assert len(out_dict["age"]) <= len(test_case_4["dict_final_pop"]["age"])
