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

import pypopsyn.simulator.multiband_emission.emission_xray as xem

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "Lx": np.array([3.0e35, 2.0e34, 2.0e33]),
        "E": np.logspace(np.log10(10), np.log10(20000), 4),
        "T": np.array([1.0e6, 2.0e6]),
        "x": np.linspace(-1.0, 1.0, 3),
        "omega": np.logspace(16, 19, 4),
        "omega_0": np.logspace(16, 19, 4),
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
                [4.09630448e14, 3.04103266e16, 2.01560767e12, 6.46295939e-78],
                [8.43731671e14, 9.35795900e16, 2.01607344e16, 1.61432375e-27],
            ]
        ),
        "dirac_delta_expected": np.array([0.0, 2.0e-17, 0.0]),
        "n_plus_expected": np.array(
            [
                [
                    [1.21306132e-17, 0.0, 0.0, 0.0],
                    [0.0, 1.21306132e-17, 0.0, 0.0],
                    [0.0, 0.0, 1.21306132e-17, 0.0],
                    [0.0, 0.0, 0.0, 1.21306132e-17],
                ],
                [
                    [1.64169997e-18, 0.0, 0.0, 0.0],
                    [0.0, 1.64169997e-18, 0.0, 0.0],
                    [0.0, 0.0, 1.64169997e-18, 0.0],
                    [0.0, 0.0, 0.0, 1.64169997e-18],
                ],
                [
                    [1.34758940e-19, 0.0, 0.0, 0.0],
                    [0.0, 1.34758940e-19, 0.0, 0.0],
                    [0.0, 0.0, 1.34758940e-19, 0.0],
                    [0.0, 0.0, 0.0, 1.34758940e-19],
                ],
            ]
        ),
        "n_minus_expected": np.array(
            [
                [
                    [9.59884715e-17, 0.0, 0.0, 0.0],
                    [0.0, 9.59884715e-18, 0.0, 0.0],
                    [0.0, 0.0, 9.59884715e-19, 0.0],
                    [0.0, 0.0, 0.0, 9.59884715e-20],
                ],
                [
                    [4.65830175e-16, 0.0, 0.0, 0.0],
                    [0.0, 4.65830175e-17, 0.0, 0.0],
                    [0.0, 0.0, 4.65830175e-18, 0.0],
                    [0.0, 0.0, 0.0, 4.65830175e-19],
                ],
                [
                    [
                        4.74303735e-15,
                        2.60148965e-18,
                        2.63269354e-20,
                        1.76972659e-21,
                    ],
                    [0.0, 4.74303735e-16, 2.60148965e-19, 2.63269354e-21],
                    [0.0, 0.0, 4.74303735e-17, 2.60148965e-20],
                    [0.0, 0.0, 0.0, 4.74303735e-18],
                ],
            ]
        ),
        "p_trans_expected": np.array(
            [
                [
                    [1.51986605e-17, 0.0, 0.0, 0.0],
                    [0.0, 1.38169641e-18, 0.0, 0.0],
                    [0.0, 0.0, 1.38169641e-19, 0.0],
                    [0.0, 0.0, 0.0, 1.51986605e-19],
                ],
                [
                    [1.11859772e-17, 0.0, 0.0, 0.0],
                    [0.0, 1.01690702e-18, 0.0, 0.0],
                    [0.0, 0.0, 1.01690702e-19, 0.0],
                    [0.0, 0.0, 0.0, 1.11859772e-19],
                ],
                [
                    [1.11116156e-17, 0.0, 0.0, 0.0],
                    [0.0, 1.01014687e-18, 0.0, 0.0],
                    [0.0, 0.0, 1.01014687e-19, 0.0],
                    [0.0, 0.0, 0.0, 1.11116156e-19],
                ],
            ]
        ),
        "p_refl_expected": np.array(
            [
                [
                    [7.02356176e-18, 0.0, 0.0, 0.0],
                    [0.0, 6.38505615e-19, 0.0, 0.0],
                    [0.0, 0.0, 6.38505615e-20, 0.0],
                    [0.0, 0.0, 0.0, 7.02356176e-20],
                ],
                [
                    [1.10362450e-17, 0.0, 0.0, 0.0],
                    [0.0, 1.00329500e-18, 0.0, 0.0],
                    [0.0, 0.0, 1.00329500e-19, 0.0],
                    [0.0, 0.0, 0.0, 1.10362450e-19],
                ],
                [
                    [
                        1.11106067e-17,
                        5.53725023e-21,
                        5.60335995e-23,
                        4.12047288e-23,
                    ],
                    [0.0, 1.00955177e-18, 5.53694635e-22, 6.12972784e-23],
                    [0.0, 0.0, 1.00949636e-19, 6.05707549e-22],
                    [0.0, 0.0, 0.0, 1.10432634e-19],
                ],
            ]
        ),
        "rcs_spectrum_expected": np.array(
            [
                [2.87907662e26, 3.83876882e27, 1.91938441e28, 1.91938441e22],
                [8.36600776e25, 1.67320155e27, 8.36600776e27, 8.36600776e21],
                [4.32869431e26, 5.04811927e27, 2.49942977e28, 2.49448275e22],
            ]
        ),
        "beta_T_expected": np.array([0.001, 0.3, 0.3]),
        "tau_res_expected": np.array([0.001, 1.0, 10.0]),
        "flux_expected": np.array(
            [6.35594512e-11, 1.19205549e-12, 6.42424218e-14]
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


def test_dirac_delta(test_case_1):
    """
    Verifying that for a given temperature the black-body intensity spectrum is correctly calculated.
    """

    dirac_delta_out = xem.dirac_delta(test_case_1["x"])

    assert np.isclose(
        test_case_1["dirac_delta_expected"],
        dirac_delta_out,
        rtol=TOL,
        atol=1.0e-23,
    ).all()


def test_n_plus(test_case_1):
    """
    Verifying that the transmission function n+ is correctly calculated.
    """

    n_plus_out = xem.n_plus(
        test_case_1["omega"],
        test_case_1["omega_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["n_plus_expected"],
        n_plus_out,
        rtol=TOL,
        atol=1.0e-23,
    ).all()


def test_n_minus(test_case_1):
    """
    Verifying that the reflection function n- is correctly calculated.
    """

    n_minus_out = xem.n_minus(
        test_case_1["omega"],
        test_case_1["omega_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["n_minus_expected"],
        n_minus_out,
        rtol=TOL,
        atol=1.0e-23,
    ).all()


def test_trans_reflect_prob(test_case_1):
    """
    Verifying that the transmission and reflection probabilities are correctly calculated.
    """

    p_trans_out, p_refl_out = xem.trans_reflect_prob(
        test_case_1["omega"],
        test_case_1["omega_0"],
        test_case_1["tau_0"],
        test_case_1["beta_T"],
    )

    assert np.isclose(
        test_case_1["p_trans_expected"],
        p_trans_out,
        rtol=TOL,
        atol=1.0e-23,
    ).all()
    assert np.isclose(
        test_case_1["p_refl_expected"],
        p_refl_out,
        rtol=TOL,
        atol=1.0e-23,
    ).all()


def test_resonant_cyclothron_scat_spectrum(test_case_1):
    """
    Verifying that the RCS spectrum is correctly calculated.
    """

    rcs_spectrum_out = xem.resonant_cyclothron_scat_spectrum(
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
    print(flux_out)

    assert np.isclose(
        test_case_1["flux_expected"],
        flux_out,
        rtol=TOL,
        atol=1.0e-14,
    ).all()
