import numpy as np
import pytest

import pypopsyn.simulator.initial_position as ip

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.5,
        "arm_index": 3,
        "theta_no_noise_expected": -1.69864,
        "r_with_noise_expected": 1.6,
        "theta_with_noise_expected": -0.69864,
        "z": 0.01,
        "p_z_expected": 9.04837,
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "NS_number": 5,
        "z": np.array([0.2, 0.3, 0.1, 0.5, 0.1]),
        "up_down_index_mock": np.array([0, 1, 0, 1, 0]),
        "z_expected": np.array([0.2, -0.3, 0.1, -0.5, 0.1]),
    }

    return data


def test_check_arm_index_01():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    arm_index = -1
    with pytest.raises(ValueError, match="Arm index is out of range"):
        ip.check_arm_index(arm_index)


def test_check_arm_index_02():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    arm_index = 6
    with pytest.raises(ValueError, match="Arm index is out of range"):
        ip.check_arm_index(arm_index)


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
        test_case_1["r"], test_case_1["arm_index"]
    )

    assert np.abs(test_case_1["theta_with_noise_expected"] - theta_out) < TOL
    assert np.abs(test_case_1["r_with_noise_expected"] - r_out) < TOL


def test_calculate_theta(test_case_1):
    """
    Verifying that the angular coordinate theta is correctly calculated.
    """
    theta_out = ip.calculate_theta(test_case_1["r"], test_case_1["arm_index"])

    assert np.abs(test_case_1["theta_no_noise_expected"] - theta_out) < TOL


def test_pdf_initial_height(test_case_1):
    """
    Verifying that distribution of stars away from the galactic plane is
    correctly calculated.
    """
    p_z_out = ip.pdf_initial_height(test_case_1["z"])
    assert np.abs(test_case_1["p_z_expected"] - p_z_out) < TOL


def test_random_scatter_about_plane_01(test_case_2):
    """
    Verifying that a ValueError is raised when the input array does not have the same
    length as the number of neutron stars simulated
    """
    NS_number = 10
    with pytest.raises(ValueError, match="Input array has the wrong length"):
        ip.random_scatter_about_plane(test_case_2["z"], NS_number)


def test_random_scatter_about_plane_02(monkeypatch, test_case_2):
    """
    Verifying that height values are correctly scattered about the z=0 axis given a
    specific up_down_index array.
    """

    def mock_index(*args, **kwargs):
        return np.array([0, 1, 0, 1, 0])

    monkeypatch.setattr(np.random, "randint", mock_index)

    z_out = ip.random_scatter_about_plane(
        test_case_2["z"], test_case_2["NS_number"]
    )

    assert np.isclose(test_case_2["z_expected"], z_out).all()
