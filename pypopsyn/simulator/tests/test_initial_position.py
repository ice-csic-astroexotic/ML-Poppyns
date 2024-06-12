"""
    Tests for the stellar_dynamics/initial_position module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.stellar_dynamics.initial_position as ip
from pypopsyn.simulator.config_simulator import cfg

TOL = 1e-5


# For the tests, set the spiral arm pattern in the configuration file to
# the one from Faucher-Giguère & Kaspi (2006).
cfg["spiral_arms"] = "saFK06"


@pytest.fixture()
def test_case_1():
    data = {
        "r": np.array([1.0]),
        "pdf_r_YK04_expected": 362.95883,
        "pdf_r_VV21_expected": 54.075275,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "NS_number": 2,
        "r": np.array([1.5, 3.0]),
        "arm_index": np.array([3, 1]),
        "phi_no_noise_expected": np.array([-1.69863, 0.93921]),
        "r_with_noise_expected": np.array([1.6, 3.2]),
        "phi_with_noise_expected": np.array([-0.69863, 2.4392]),
        "z": 0.01,
        "pdf_z_expected": 5.25533,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "NS_number": 5,
        "z": np.array([0.2, 0.3, 0.1, 0.5, 0.1]),
        "if_below_mock": np.array([False, True, False, True, False]),
        "z_expected": np.array([0.2, -0.3, 0.1, -0.5, 0.1]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "NS_number": 2,
        "r": np.array([1.2, 3.8]),
        "uniform_noise_mock": np.array([2.0, 0.2]),
        "phi_corr_expected": np.array([1.31409, 0.052895]),
        "r_corr_expected": np.array([0.02, 0.5]),
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "phi0": np.array([0.0, 1.0]),
        "t": np.array([1.0e8, 1e4]),
        "phi_t_expected": np.array([2.51327, 1.00025]),
    }

    return data


def test_pdf_radial_density_YK04(test_case_1):
    """
    Verifying that the pdf for the pulsar radial density from Yusifov & Küçük (2004) is correctly calculated.
    """
    pdf_r_out = ip.pdf_radial_density_YK04(test_case_1["r"])
    assert np.abs(pdf_r_out - test_case_1["pdf_r_YK04_expected"]) < TOL


def test_pdf_radial_density_VV21(test_case_1):
    """
    Verifying that the pdf for the pulsar radial density from Verberne & Vink (2021) is correctly calculated.
    """
    pdf_r_out = ip.pdf_radial_density_VV21(test_case_1["r"])
    assert np.abs(pdf_r_out - test_case_1["pdf_r_VV21_expected"]) < TOL


def test_smear_initial_coordinates(monkeypatch, test_case_2):
    """
    Verifying that for a given choice of noise in galactocentric coordinates
    the resulting phi and r values are correctly calculated.
    """
    # Mocking the noise parameters that are otherwise randomly determined;
    # return is the same order as the original function, i.e., phi_corr, r_corr.

    def mock_noise(*args, **kwargs):
        return np.array([1.0, 1.5]), np.array([0.1, 0.2])

    monkeypatch.setattr(ip, "calculate_noise_for_coordinates", mock_noise)

    phi_out, r_out = ip.smear_initial_coordinates(
        test_case_2["r"],
        test_case_2["phi_no_noise_expected"],
        test_case_2["NS_number"],
    )

    assert np.isclose(
        test_case_2["phi_with_noise_expected"], phi_out, rtol=TOL, atol=1.0e-30
    ).all()
    assert np.isclose(
        test_case_2["r_with_noise_expected"], r_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_spiral_arm_time_evol(test_case_5):
    """
    Verifying that the spiral structure evolves in time in the correct way.
    """

    phi_t_out = ip.spiral_arm_time_evol(test_case_5["phi0"], test_case_5["t"])

    assert np.isclose(
        test_case_5["phi_t_expected"], phi_t_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_calculate_noise_for_coordinates(monkeypatch, test_case_4):
    """
    Verifying that the noise is correctly calculated.
    """

    def mock_noise_uniform(*args, **kwargs):
        return test_case_4["uniform_noise_mock"]

    monkeypatch.setattr(np.random, "uniform", mock_noise_uniform)

    def mock_noise_normal(*args, **kwargs):
        return test_case_4["r_corr_expected"]

    monkeypatch.setattr(np.random, "normal", mock_noise_normal)

    phi_corr_out, r_corr_out = ip.calculate_noise_for_coordinates(
        test_case_4["r"], test_case_4["NS_number"]
    )

    assert np.isclose(
        test_case_4["phi_corr_expected"], phi_corr_out, rtol=TOL, atol=1.0e-30
    ).all()
    assert np.isclose(
        test_case_4["r_corr_expected"], r_corr_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_pdf_initial_height(test_case_2):
    """
    Verifying that distribution of stars away from the galactic plane is
    correctly calculated.
    """
    pdf_z_out = ip.pdf_initial_height(test_case_2["z"])
    assert np.abs(test_case_2["pdf_z_expected"] - pdf_z_out) < TOL


def test_random_scatter_about_plane_01(test_case_3):
    """
    Verifying that a ValueError is raised when the input array does not have the same
    length as the number of neutron stars simulated.
    """
    NS_number = 10
    with pytest.raises(ValueError, match="Input array has the wrong length"):
        ip.random_scatter_about_plane(test_case_3["z"], NS_number)


def test_random_scatter_about_plane_02(monkeypatch, test_case_3):
    """
    Verifying that height values are correctly scattered about the z=0 axis given a
    specific up_down_index array.
    """

    def mock_index(*args, **kwargs):
        return test_case_3["if_below_mock"]

    monkeypatch.setattr(np.random, "choice", mock_index)

    z_out = ip.random_scatter_about_plane(
        test_case_3["z"], test_case_3["NS_number"]
    )

    assert np.isclose(test_case_3["z_expected"], z_out).all()
