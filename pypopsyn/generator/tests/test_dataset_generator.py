"""
Test for the dataset_generator module

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright(c) MAGNESIA(ICE - CSIC)
"""

import pytest

import pypopsyn.generator.dataset_generator as dg


@pytest.fixture()
def test_case_1():
    data = {
        "dict1": {"p1": 1, "p2": 2},
        "dict2": {"p1": 3, "p2": 4},
        "dict_expected": {"p1": [1, 3], "p2": [2, 4]},
    }

    return data


def test_merge_dict(test_case_1):
    """
        Checking that dictionaries are merged in the correct way
    """
    dict_out = dg.merge_dict(test_case_1["dict1"], test_case_1["dict2"])

    assert test_case_1["dict_expected"] == dict_out
