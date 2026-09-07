"""
    Test for the import_param_statistics.py module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import json

import numpy as np
import pytest

from mlpoppyns.learning.utils.import_param_statistics import import_statistics


@pytest.fixture
def test_case_1():
    data = {
        "stats": {
            "parameter_1": {
                "mean": 1.0,
                "std": 0.1,
                "min": 0.5,
                "max": 1.5,
            },
            "parameter_2": {
                "mean": 2.0,
                "std": 0.2,
                "min": 1.5,
                "max": 2.5,
            },
            "parameter_3": {
                "mean": 3.0,
                "std": 0.3,
                "min": 2.5,
                "max": 3.5,
            },
        },
        "mean_list_expected": np.array([1.0, 2.0, 3.0]),
        "std_list_expected": np.array([0.1, 0.2, 0.3]),
        "min_list_expected": np.array([0.5, 1.5, 2.5]),
        "max_list_expected": np.array([1.5, 2.5, 3.5]),
    }

    return data


def test_import_statistics(tmp_path, test_case_1):
    """
    Test that statistics are correctly imported from a JSON file.
    """

    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(test_case_1["stats"]))

    mean, std, max_values, min_values = import_statistics(str(stats_file))

    np.testing.assert_array_equal(mean, test_case_1["mean_list_expected"])
    np.testing.assert_array_equal(std, test_case_1["std_list_expected"])
    np.testing.assert_array_equal(max_values, test_case_1["max_list_expected"])
    np.testing.assert_array_equal(min_values, test_case_1["min_list_expected"])
