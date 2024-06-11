"""
    Test for the axes_scaling.py module

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.generator.maps.axes_scaling as axs


@pytest.fixture()
def test_case_1():
    data = {
        "x_range": (1.0e-3, 10.0),
        "y_range": (1.0e-2, 100.0),
        "x_log_scale": True,
        "y_log_scale": True,
        "n_x_bins": 4,
        "n_y_bins": 4,
        "x_edges_expected": np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]),
        "y_edges_expected": np.array([1.0e-2, 1.0e-1, 1, 10.0, 100.0]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "x_range": (0.0, 10.0),
        "y_range": (0.0, 100.0),
        "x_log_scale": False,
        "y_log_scale": False,
        "n_x_bins": 5,
        "n_y_bins": 5,
        "x_edges_expected": np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        "y_edges_expected": np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0]),
    }

    return data


def test_check_range_above_zero_01():
    """
    Verifying that a ValueError is raised if the range is not above zero.
    """
    r_range = (0.0, 0.1)
    with pytest.raises(ValueError, match="Value range has to be above zero."):
        axs.check_range_above_zero(r_range)


def test_check_range_above_zero_02():
    """
    Verifying that a ValueError is raised if the range is not above zero.
    """
    r_range = (1.0, -0.1)
    with pytest.raises(ValueError, match="Value range has to be above zero."):
        axs.check_range_above_zero(r_range)


def test_log_scale_vs_linear_scale_01(test_case_1):
    """
    Verifying bin edges are correctly calculated for logarithmic x and y axes.
    """
    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        test_case_1["x_range"],
        test_case_1["y_range"],
        test_case_1["x_log_scale"],
        test_case_1["y_log_scale"],
        test_case_1["n_x_bins"],
        test_case_1["n_y_bins"],
    )
    assert np.isclose(x_edges, test_case_1["x_edges_expected"]).all()
    assert np.isclose(y_edges, test_case_1["y_edges_expected"]).all()


def test_log_scale_vs_linear_scale_02(test_case_2):
    """
    Verifying bin edges are correctly calculated for linear x and y axes.
    """
    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        test_case_2["x_range"],
        test_case_2["y_range"],
        test_case_2["x_log_scale"],
        test_case_2["y_log_scale"],
        test_case_2["n_x_bins"],
        test_case_2["n_y_bins"],
    )
    assert np.isclose(x_edges, test_case_2["x_edges_expected"]).all()
    assert np.isclose(y_edges, test_case_2["y_edges_expected"]).all()


def test_log_scale_vs_linear_scale_03(test_case_1, test_case_2):
    """
    Verifying bin edges are correctly calculated for linear x and logarithmic y axes.
    """
    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        test_case_2["x_range"],
        test_case_1["y_range"],
        test_case_2["x_log_scale"],
        test_case_1["y_log_scale"],
        test_case_2["n_x_bins"],
        test_case_1["n_y_bins"],
    )
    assert np.isclose(x_edges, test_case_2["x_edges_expected"]).all()
    assert np.isclose(y_edges, test_case_1["y_edges_expected"]).all()


def test_log_scale_vs_linear_scale_04(test_case_1, test_case_2):
    """
    Verifying bin edges are correctly calculated for logarithmic x and linear y axes.
    """
    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        test_case_1["x_range"],
        test_case_2["y_range"],
        test_case_1["x_log_scale"],
        test_case_2["y_log_scale"],
        test_case_1["n_x_bins"],
        test_case_2["n_y_bins"],
    )
    assert np.isclose(x_edges, test_case_1["x_edges_expected"]).all()
    assert np.isclose(y_edges, test_case_2["y_edges_expected"]).all()
