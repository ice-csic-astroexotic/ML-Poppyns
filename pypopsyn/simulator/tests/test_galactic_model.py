import numpy as np
import pytest

import pypopsyn.simulator.galactic_model as gm

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.0,
        "z": 1.0,
        "K_expected": 3.42339,
        "dK_dz_expected": 0.97763,
        "dpot_dh_dr_expected": 2.32457e-15,
        "dpot_dh_dz_expected": 7.77990e-15,
        "dpot_b_dr_expected": 3.83448e-14,
        "dpot_n_dr_expected": 7.70712e-15,
        "gradient_mw_expected": np.array([4.83765e-14, 0.0, 7.77990e-15]),
    }

    return data


def test_shape_parameter(test_case_1):
    """
    Verifying that the shape parameter and its derivative are evaluated correctly
    """
    K_out, dK_dz_out = gm.shape_parameter(test_case_1["z"])

    assert np.abs(test_case_1["K_expected"] - K_out) < TOL
    assert np.abs(test_case_1["dK_dz_expected"] - dK_dz_out) < TOL


def test_r_z_derivative_dh_potential(test_case_1):
    """
    Verifying that the r and z derivative of the disk-halo potential are evaluated
    correctly
    """
    dpot_dh_dr_out, dpot_dh_dz_out = gm.r_z_derivatives_dh_potential(
        test_case_1["r"], test_case_1["z"]
    )

    assert np.isclose(
        dpot_dh_dr_out,
        test_case_1["dpot_dh_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_dh_dz_out,
        test_case_1["dpot_dh_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_derivative_b_potential(test_case_1):
    """
    Verifying that the radial derivative of the bulge potential is evaluated
    correctly
    """
    dpot_b_dr_out = gm.r_derivative_b_potential(test_case_1["r"])

    assert np.isclose(
        dpot_b_dr_out,
        test_case_1["dpot_b_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_derivative_n_potential(test_case_1):
    """
    Verifying that the radial derivative of the nucleus potential is evaluated
    correctly
    """
    dpot_n_dr_out = gm.r_derivative_n_potential(test_case_1["r"])

    assert np.isclose(
        dpot_n_dr_out,
        test_case_1["dpot_n_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_cylind_coord_gradient_mw_potential(test_case_1):
    """
    Verifying that the gradient in cylindrical coordinates of the Milky Way potential is evaluated
    correctly
    """
    gradient_mw_out = gm.cylind_coord_gradient_mw_potential(
        test_case_1["r"], test_case_1["z"]
    )

    assert np.isclose(
        gradient_mw_out,
        test_case_1["gradient_mw_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
