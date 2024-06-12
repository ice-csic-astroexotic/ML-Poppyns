"""
    Tests for the stellar_dynamics/galactic_model module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.simulator.stellar_dynamics.galactic_model as gm

TOL = 1e-5

gmFK06 = gm.GalaxyModelFK06()
gmM19 = gm.GalaxyModelM19()


@pytest.fixture()
def test_case_1():
    data = {
        "r": 1.0,
        "z": 1.0,
        "r_array": np.array([1.0, 10.0]),
        "z_array": np.array([1.0, 2.0]),
        "v_array": np.array([100.0, 1000.0]),
        "K_expected": 3.42339,
        "dK_dz_expected": 0.97763,
        "pot_dh_expected": -9.56297e14,
        "pot_b_expected": -2.09706e14,
        "pot_n_expected": -2.799570e14,
        "pot_MW_expected": -1.44596e15,
        "tot_energy_expected": 3.01037e15,
        "tot_angular_momentum_expected": 3.11653e30,
        "dpot_dh_dr_expected": 2.32457e-15,
        "dpot_dh_dz_expected": 7.77990e-15,
        "dpot_b_dr_expected": 5.153876e-15,
        "dpot_b_dz_expected": 5.153876e-15,
        "dpot_n_dr_expected": 1.417782e-14,
        "dpot_n_dz_expected": 1.417782e-14,
        "gradient_mw_expected": np.array([2.165627e-14, 0.0, 2.711159e-14]),
    }
    return data


@pytest.fixture()
def test_case_2():
    data = {
        "r": 1.0,
        "z": 1.0,
        "r_array": np.array([1.0, 10.0]),
        "z_array": np.array([1.0, 2.0]),
        "v_array": np.array([100.0, 1000.0]),
        "K_expected": 4.03846,
        "dK_dz_expected": 0.96296,
        "pot_d_expected": -7.06604e14,
        "pot_b_expected": -8.95364e13,
        "pot_n_expected": -4.98087e13,
        "pot_h_expected": -1.43074e15,
        "pot_MW_expected": -2.27669e15,
        "tot_energy_expected": 1.33370e15,
        "dpot_d_dr_expected": 4.26395e-15,
        "dpot_d_dz_expected": 1.65820e-14,
        "dpot_b_dr_expected": 2.73918e-15,
        "dpot_b_dz_expected": 2.73918e-15,
        "dpot_n_dr_expected": 2.47859e-15,
        "dpot_n_dz_expected": 2.47859e-15,
        "dpot_h_dr_expected": 3.14655e-15,
        "dpot_h_dz_expected": 3.14655e-15,
        "gradient_mw_expected": np.array([1.26283e-14, 0.0, 2.49464e-14]),
    }
    return data


def test_shape_parameter_FK06(test_case_1):
    """
    Verifying that the shape parameter and its derivative are evaluated correctly.
    """
    K_out, dK_dz_out = gmFK06.shape_parameter(test_case_1["z"])

    assert np.abs(test_case_1["K_expected"] - K_out) < TOL
    assert np.abs(test_case_1["dK_dz_expected"] - dK_dz_out) < TOL


def test_dh_potential_FK06(test_case_1):
    """
    Verifying that the disk-halo potential is evaluated correctly.
    """
    pot_dh_out = gmFK06.dh_potential(test_case_1["r"], test_case_1["z"])

    assert np.isclose(
        pot_dh_out,
        test_case_1["pot_dh_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_b_potential_FK06(test_case_1):
    """
    Verifying that the bulge potential is evaluated correctly.
    """
    pot_b_out = gmFK06.b_potential(test_case_1["r"], test_case_1["z"])
    assert np.isclose(
        pot_b_out,
        test_case_1["pot_b_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_n_potential_FK06(test_case_1):
    """
    Verifying that the nucleus potential is evaluated correctly.
    """
    pot_n_out = gmFK06.n_potential(test_case_1["r"], test_case_1["z"])

    assert np.isclose(
        pot_n_out,
        test_case_1["pot_n_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_MW_potential_FK06(test_case_1):
    """
    Verifying that the total Milky Way potential is evaluated correctly.
    """
    pot_MW_out = gmFK06.MW_potential(test_case_1["r"], test_case_1["z"])

    assert np.isclose(
        pot_MW_out,
        test_case_1["pot_MW_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_tot_energy_FK06(test_case_1):
    """
    Verifying that the total energy is evaluated correctly.
    """
    tot_energy_out = gmFK06.total_energy(
        test_case_1["v_array"], test_case_1["r_array"], test_case_1["z_array"]
    )

    assert np.isclose(
        tot_energy_out,
        test_case_1["tot_energy_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_tot_angular_momentum(test_case_1):
    """
    Verifying that the z-component of the total angular momentum is evaluated correctly.
    """
    L_z_out = gmFK06.total_angular_momentum_z(
        test_case_1["v_array"], test_case_1["r_array"]
    )

    assert np.isclose(
        L_z_out,
        test_case_1["tot_angular_momentum_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivative_dh_potential_FK06(test_case_1):
    """
    Verifying that the r and z derivatives of the disk-halo potential are evaluated
    correctly.
    """
    dpot_dh_dr_out, dpot_dh_dz_out = gmFK06.r_z_derivatives_dh_potential(
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


def test_r_z_derivative_b_potential_FK06(test_case_1):
    """
    Verifying that the radial and z derivatives of the bulge potential is evaluated
    correctly.
    """
    dpot_b_dr_out, dpot_b_dz_out = gmFK06.r_z_derivatives_b_potential(
        test_case_1["r"], test_case_1["z"]
    )

    assert np.isclose(
        dpot_b_dr_out,
        test_case_1["dpot_b_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )

    assert np.isclose(
        dpot_b_dz_out,
        test_case_1["dpot_b_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivative_n_potential_FK06(test_case_1):
    """
    Verifying that the radial and z derivatives of the nucleus potential is evaluated
    correctly.
    """
    dpot_n_dr_out, dpot_n_dz_out = gmFK06.r_z_derivatives_n_potential(
        test_case_1["r"], test_case_1["z"]
    )

    assert np.isclose(
        dpot_n_dr_out,
        test_case_1["dpot_n_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_n_dr_out,
        test_case_1["dpot_n_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_cylind_coord_gradient_mw_potential_FK06(test_case_1):
    """
    Verifying that the gradient in cylindrical coordinates of the Milky Way
    potential is evaluated correctly.
    """
    gradient_mw_out = gmFK06.cylind_coord_gradient_mw_potential(
        test_case_1["r"], test_case_1["z"]
    )

    assert np.isclose(
        gradient_mw_out,
        test_case_1["gradient_mw_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_shape_parameter_M19(test_case_2):
    """
    Verifying that the shape parameter and its derivative are evaluated correctly.
    """
    K_out, dK_dz_out = gmM19.shape_parameter(test_case_2["z"])

    assert np.abs(test_case_2["K_expected"] - K_out) < TOL
    assert np.abs(test_case_2["dK_dz_expected"] - dK_dz_out) < TOL


def test_d_potential_M19(test_case_2):
    """
    Verifying that the disk potential is evaluated correctly.
    """
    pot_d_out = gmM19.d_potential(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        pot_d_out,
        test_case_2["pot_d_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_b_potential_M19(test_case_2):
    """
    Verifying that the bulge potential is evaluated correctly.
    """
    pot_b_out = gmM19.b_potential(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        pot_b_out,
        test_case_2["pot_b_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_n_potential_M19(test_case_2):
    """
    Verifying that the nucleus potential is evaluated correctly.
    """
    pot_n_out = gmM19.n_potential(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        pot_n_out,
        test_case_2["pot_n_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_h_potential_M19(test_case_2):
    """
    Verifying that the halo potential is evaluated correctly.
    """
    pot_h_out = gmM19.h_potential(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        pot_h_out,
        test_case_2["pot_h_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_MW_potential_M19(test_case_2):
    """
    Verifying that the total Milky Way potential is evaluated correctly.
    """
    pot_MW_out = gmM19.MW_potential(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        pot_MW_out,
        test_case_2["pot_MW_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_tot_energy_M19(test_case_2):
    """
    Verifying that the total energy is evaluated correctly.
    """
    tot_energy_out = gmM19.total_energy(
        test_case_2["v_array"], test_case_2["r_array"], test_case_2["z_array"]
    )

    assert np.isclose(
        tot_energy_out,
        test_case_2["tot_energy_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivative_d_potential_M19(test_case_2):
    """
    Verifying that the r and z derivatives of the disk-halo potential are evaluated
    correctly.
    """
    dpot_d_dr_out, dpot_d_dz_out = gmM19.r_z_derivatives_d_potential(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        dpot_d_dr_out,
        test_case_2["dpot_d_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_d_dz_out,
        test_case_2["dpot_d_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivatives_b_potential_M19(test_case_2):
    """
    Verifying that the radial and z derivatives of the bulge potential is evaluated
    correctly.
    """
    dpot_b_dr_out, dpot_b_dz_out = gmM19.r_z_derivatives_b_potential(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        dpot_b_dr_out,
        test_case_2["dpot_b_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_b_dz_out,
        test_case_2["dpot_b_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivatives_n_potential_M19(test_case_2):
    """
    Verifying that the radial and z derivatives of the nucleus potential is evaluated
    correctly.
    """
    dpot_n_dr_out, dpot_n_dz_out = gmM19.r_z_derivatives_n_potential(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        dpot_n_dr_out,
        test_case_2["dpot_n_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_n_dz_out,
        test_case_2["dpot_n_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_r_z_derivatives_h_potential_M19(test_case_2):
    """
    Verifying that the radial and z derivatives of the halo potential is evaluated
    correctly.
    """
    dpot_h_dr_out, dpot_h_dz_out = gmM19.r_z_derivatives_h_potential(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        dpot_h_dr_out,
        test_case_2["dpot_h_dr_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
    assert np.isclose(
        dpot_h_dz_out,
        test_case_2["dpot_h_dz_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )


def test_cylind_coord_gradient_mw_potential_M19(test_case_2):
    """
    Verifying that the gradient in cylindrical coordinates of the Milky Way
    potential is evaluated correctly.
    """
    gradient_mw_out = gmM19.cylind_coord_gradient_mw_potential(
        test_case_2["r"], test_case_2["z"]
    )

    assert np.isclose(
        gradient_mw_out,
        test_case_2["gradient_mw_expected"],
        rtol=TOL,
        atol=1.0e-30,
    ).all()
