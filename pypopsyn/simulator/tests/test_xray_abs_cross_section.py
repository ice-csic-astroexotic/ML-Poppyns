"""
    Tests for the xray_abs_cross_section.py module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.interstellar_medium.xray_abs_cross_section as xabs

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "energy": np.array([50.0, 1000.0, 5000.0, 9000.0]),
        "wavelength": np.array([0.5, 1.5, 2.5]),
        "q": 2.5,
        "nu": 3,
        "gamma": 1.0e-3,
        "abundances": np.array(
            [
                12.0,
                11.0,
                8.65,
                7.96,
                8.87,
                8.14,
                6.32,
                7.60,
                6.49,
                7.57,
                7.28,
                5.28,
                6.58,
                6.35,
                5.69,
                7.52,
                6.26,
            ]
        ),
        "fano_line_expected": np.array([0.99999862, 0.99999585, 0.99999305]),
        "al_mass_abs_coeff_expected": np.array(
            [9491.40494049, 1170.1673389, 187.41536837, 33.47134242]
        ),
        "ar_mass_abs_coeff_expected": np.array(
            [11595.09363726, 3404.34165175, 438.6967572, 86.04135385]
        ),
        "ca_mass_abs_coeff_expected": np.array(
            [15268.0851086, 4778.43892658, 620.2913202, 125.58810757]
        ),
        "c_mass_abs_coeff_expected": np.array(
            [91709.8454, 2261.82673, 18.3141387, 3.02094648]
        ),
        "chl_mass_abs_coeff_expected": np.array(
            [21379.46693877, 2802.30122954, 389.35737306, 74.86739893]
        ),
        "chr_mass_abs_coeff_expected": np.array(
            [81862.23427677, 7428.14024288, 102.51315819, 181.94285631]
        ),
        "he_mass_abs_coeff_expected": np.array(
            [302178.178, 87.6804353, 0.834151945, 0.119010962]
        ),
        "h_mass_abs_coeff_expected": np.array(
            [94123.9589, 6.78960273, 2.92526084e-02, 3.92302480e-03]
        ),
        "fe_mass_abs_coeff_expected": np.array(
            [89636.90480611, 8984.94027122, 134.67869363, 225.53241102]
        ),
        "mg_mass_abs_coeff_expected": np.array(
            [105627.947, 919.322347, 157.775607, 27.9666588]
        ),
        "ne_mass_abs_coeff_expected": np.array(
            [241036.967, 7659.05509, 92.0361941, 15.6066549]
        ),
        "ni_mass_abs_coeff_expected": np.array(
            [86783.10046349, 10812.31596659, 175.24148819, 272.36207258]
        ),
        "n_mass_abs_coeff_expected": np.array(
            [163625.242, 3446.40881, 29.1336770, 5.04690575]
        ),
        "o_mass_abs_coeff_expected": np.array(
            [218444.592, 4607.14381, 48.0255736, 7.74001907]
        ),
        "si_mass_abs_coeff_expected": np.array(
            [14892.18319477, 1623.21064176, 237.61578225, 43.37146902]
        ),
        "so_mass_abs_coeff_expected": np.array(
            [184304.079, 601.311708, 114.877538, 19.8192073]
        ),
        "s_mass_abs_coeff_expected": np.array(
            [22137.92829046, 2515.96709403, 343.81230461, 65.0634517]
        ),
        "cross_section_tot_expected": np.array(
            [3.65505014e-19, 2.61754149e-22, 3.90567519e-24, 1.33009169e-24]
        ),
        "cross_section_approx_expected": np.array(
            [3.38640000e-19, 2.42200000e-22, 3.52520000e-24, 1.27297668e-24]
        ),
    }

    return data


def test_aluminium(test_case_1):
    """
    Verifying that the mass absorption coefficient for aluminium is computed correctly.
    """

    mass_abs_coeff = xabs.aluminium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["al_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_argon(test_case_1):
    """
    Verifying that the mass absorption coefficient for argon is computed correctly.
    """

    mass_abs_coeff = xabs.argon(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["ar_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_calcium(test_case_1):
    """
    Verifying that the mass absorption coefficient for aluminium is computed correctly.
    """

    mass_abs_coeff = xabs.calcium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["ca_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_carbon(test_case_1):
    """
    Verifying that the mass absorption coefficient for aluminium is computed correctly.
    """

    mass_abs_coeff = xabs.carbon(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["c_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_chlorine(test_case_1):
    """
    Verifying that the mass absorption coefficient for chlorine is computed correctly.
    """

    mass_abs_coeff = xabs.chlorine(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["chl_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_chromium(test_case_1):
    """
    Verifying that the mass absorption coefficient for chromium is computed correctly.
    """

    mass_abs_coeff = xabs.chromium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["chr_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_fano_resonance_line(test_case_1):
    """
    Verifying that the fano resonance line is computed correctly.
    """
    fano_output = xabs.fano_resonance_line(
        test_case_1["q"],
        test_case_1["nu"],
        test_case_1["gamma"],
        test_case_1["wavelength"],
    )

    assert np.isclose(
        test_case_1["fano_line_expected"],
        fano_output,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_helium(test_case_1):
    """
    Verifying that the mass absorption coefficient for helium is computed correctly.
    """

    mass_abs_coeff = xabs.helium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["he_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_hydrogen(test_case_1):
    """
    Verifying that the mass absorption coefficient for hydrogen is computed correctly.
    """

    mass_abs_coeff = xabs.hydrogen(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["h_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_iron(test_case_1):
    """
    Verifying that the mass absorption coefficient for iron is computed correctly.
    """

    mass_abs_coeff = xabs.iron(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["fe_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_magnesium(test_case_1):
    """
    Verifying that the mass absorption coefficient for magnesium is computed correctly.
    """

    mass_abs_coeff = xabs.magnesium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["mg_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_neon(test_case_1):
    """
    Verifying that the mass absorption coefficient for neon is computed correctly.
    """

    mass_abs_coeff = xabs.neon(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["ne_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_nickel(test_case_1):
    """
    Verifying that the mass absorption coefficient for nickel is computed correctly.
    """

    mass_abs_coeff = xabs.nickel(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["ni_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_nitrogen(test_case_1):
    """
    Verifying that the mass absorption coefficient for nitrogen is computed correctly.
    """

    mass_abs_coeff = xabs.nitrogen(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["n_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_oxygen(test_case_1):
    """
    Verifying that the mass absorption coefficient for oxygen is computed correctly.
    """

    mass_abs_coeff = xabs.oxygen(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["o_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_silicon(test_case_1):
    """
    Verifying that the mass absorption coefficient for silicon is computed correctly.
    """

    mass_abs_coeff = xabs.silicon(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["si_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_sodium(test_case_1):
    """
    Verifying that the mass absorption coefficient for sodium is computed correctly.
    """

    mass_abs_coeff = xabs.sodium(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["so_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_sulfur(test_case_1):
    """
    Verifying that the mass absorption coefficient for sulfur is computed correctly.
    """

    mass_abs_coeff = xabs.sulfur(
        test_case_1["energy"],
    )

    assert np.isclose(
        test_case_1["s_mass_abs_coeff_expected"],
        mass_abs_coeff,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_absorption_cross_section_tot(test_case_1):
    """
    Verifying that the total cross section is computed correctly.
    """

    cross_section = xabs.absorption_cross_section_tot(
        test_case_1["energy"], test_case_1["abundances"]
    )

    assert np.isclose(
        test_case_1["cross_section_tot_expected"],
        cross_section,
        rtol=TOL,
        atol=1.0e-5,
    ).all()


def test_absorption_cross_section_approx(test_case_1):
    """
    Verifying that the approximated cross section is computed correctly.
    """

    cross_section = xabs.absorption_cross_section_approx(test_case_1["energy"])

    assert np.isclose(
        test_case_1["cross_section_approx_expected"],
        cross_section,
        rtol=TOL,
        atol=1.0e-5,
    ).all()
