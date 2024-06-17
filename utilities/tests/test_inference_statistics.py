"""
    Tests for the inference_statistics module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import numpy as np
import pytest

import utilities.inference_statistics as stat

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "x": np.array([1.1, 1.6, 1.9, 2.6, 3.2, 3.3, 4.2]),
        "targets": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        "n_bins": 2,
        "bin_centers_expected": np.array([1.75, 3.25]),
        "running_rmse_expected": np.array([0.1, 0.2]),
        "running_average_expected": np.array([0.033333, 0.066667]),
        "running_mre_expected": np.array([0.052222, 0.057937]),
    }

    return data


def test_statistics(test_case_1):
    """
    Checking that the cdf is correctly calculated for given pdf and x array.
    """
    (
        bin_centers_out,
        running_rmse_out,
        running_average_out,
        running_mre_out,
    ) = stat.inference_running_stat(
        test_case_1["x"], test_case_1["targets"], test_case_1["n_bins"]
    )
    assert np.isclose(
        bin_centers_out, test_case_1["bin_centers_expected"]
    ).all()
    assert np.isclose(
        running_rmse_out, test_case_1["running_rmse_expected"]
    ).all()
    assert np.isclose(
        running_average_out, test_case_1["running_average_expected"]
    ).all()
    assert np.isclose(
        running_mre_out, test_case_1["running_mre_expected"]
    ).all()
