"""
Tests for the x-ray emission module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2024

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

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.multiband_emission.emission_xray as xem

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
        "n_plus_no_delta_expected": np.array(
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
        "flux_expected": np.array(
            [6.35737874e-11, 1.06565796e-12, 5.86406010e-14]
        ),
    }

    return data


def test_T_from_Lx(test_case_1):
    """
    Verifying that for a given x-ray luminosity the temperature is correctly calculated.
    """

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


def test_n_plus_no_delta(test_case_1):
    """
    Verifying that the transmission function n+ without the Dirac delta term is correctly calculated.
    """

    n_plus_no_delta_out = xem.n_plus_no_delta(
        test_case_1["E"],
        test_case_1["E_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["n_plus_no_delta_expected"],
        n_plus_no_delta_out,
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


def test_flux_xray_absorbed(test_case_1):
    """
    Verifying that absorbed X-ray flux is correctly estimated.
    """

    flux_out = xem.flux_xray_absorbed(
        test_case_1["Lx"],
        test_case_1["B"],
        test_case_1["RA"],
        test_case_1["DEC"],
        test_case_1["d"],
    )

    assert np.isclose(
        test_case_1["flux_expected"],
        flux_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()
