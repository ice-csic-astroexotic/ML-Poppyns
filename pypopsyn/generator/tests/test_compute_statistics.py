"""
    Test for the compute_statistics.py module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pytest

import pypopsyn.generator.compute_statistics as cs

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "dataset_dict": {
            "input:map1": ["map1_1", "map1_2", "map1_3", "map1_4"],
            "input:map2": ["map2_1", "map2_2", "map2_3", "map2_4"],
            "label1": [1, 2, 3, 4],
            "label2": [5, 6, 7, 8],
        },
        "statistic_dict_expected": {
            "label1": {"mean": 2.5, "std": 1.11803, "max": 4, "min": 1},
            "label2": {"mean": 6.5, "std": 1.11803, "max": 8, "min": 5},
        },
    }

    return data


def test_compute_statistics(test_case_1):
    """
    Verifying that the statistics are computed correctly and saved in the correct format.
    """
    statistic_dict_out = cs.compute_statistics(test_case_1["dataset_dict"])

    assert (
        statistic_dict_out.keys()
        == test_case_1["statistic_dict_expected"].keys()
    )

    for key1 in test_case_1["statistic_dict_expected"].keys():
        assert (
            statistic_dict_out[key1].keys()
            == test_case_1["statistic_dict_expected"][key1].keys()
        )
        for key2 in test_case_1["statistic_dict_expected"][key1].keys():
            assert (
                np.abs(
                    statistic_dict_out[key1][key2]
                    - test_case_1["statistic_dict_expected"][key1][key2]
                )
                < TOL
            )
