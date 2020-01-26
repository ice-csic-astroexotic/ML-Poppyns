import numpy as np
import pytest

import pypopsyn.simulator.initial_position as ip

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.5,
        "i": 3,
        "theta_no_noise_expected": -1.69864,
        "r_with_noise_expected": 1.6,
        "theta_with_noise_expected": -0.69864,
        "z": 0.01,
        "rho_z_expected": 9.04837,
    }

    return data


def test_check_radial_coordinate():
    """
    Verifying that a ValueError is raised if the radial coordinate is negative.
    """
    r = -0.1
    with pytest.raises(ValueError, match="Radial coordinate is out of range"):
        ip.check_radial_coordinate(r)


def test_check_arm_index_01():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    i = -1
    with pytest.raises(ValueError, match="Arm index is out of range"):
        ip.check_arm_index(i)


def test_check_arm_index_02():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    i = 6
    with pytest.raises(ValueError, match="Arm index is out of range"):
        ip.check_arm_index(i)


def test_stellar_surf_density():
    """
    Verifying that the stellar surface density is correctly calculated.
    """
    r = 1
    rho_expected = 57.76669
    rho_out = ip.stellar_surf_density(r)
    assert np.abs(rho_out - rho_expected) < TOL


def test_pdf_initial_coordinates(monkeypatch, test_case_1):
    """
    Verifying that for a given choice of noise in galactocentric coordinates
    the resulting theta and r values are correctly calculated.
    """
    # mocking the noise parameters that are otherwise randomly determined;
    # return is the same order as the original function, i.e., theta_corr, r_corr

    def mock_noise(*args, **kwargs):
        return 1.0, 0.1

    monkeypatch.setattr(ip, "calculate_noise_for_coordinates", mock_noise)

    theta_out, r_out = ip.pdf_initial_coordinates(
        test_case_1["r"], test_case_1["i"]
    )

    assert np.abs(test_case_1["theta_with_noise_expected"] - theta_out) < TOL
    assert np.abs(test_case_1["r_with_noise_expected"] - r_out) < TOL


def test_calculate_theta(test_case_1):
    """
    Verifying that the angular coordinate theta is correctly calculated.
    """
    theta_out = ip.calculate_theta(test_case_1["r"], test_case_1["i"])

    assert np.abs(test_case_1["theta_no_noise_expected"] - theta_out) < TOL


def test_pdf_initial_height(test_case_1):
    """
    Verifying that distribution of stars away from the galactic plane is
    correctly calculated.
    """
    rho_z_out = ip.pdf_initial_height(test_case_1["z"])
    assert np.abs(test_case_1["rho_z_expected"] - rho_z_out) < TOL
