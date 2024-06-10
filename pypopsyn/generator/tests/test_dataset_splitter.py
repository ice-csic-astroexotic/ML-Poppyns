"""
Test for the dataset_splitter.py module

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pytest

import pypopsyn.generator.dataset_splitter as ds

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
        "split": 0.5,
        "valid_dict_expected": {
            "input:map1": np.array(["map1_2", "map1_3"]),
            "input:map2": np.array(["map2_2", "map2_3"]),
            "label1": np.array([2, 3]),
            "label2": np.array([6, 7]),
        },
        "train_dict_expected": {
            "input:map1": np.array(["map1_1", "map1_4"]),
            "input:map2": np.array(["map2_1", "map2_4"]),
            "label1": np.array([1, 4]),
            "label2": np.array([5, 8]),
        },
    }

    return data


def test_split_dataset(monkeypatch, test_case_1):
    """
    Verifying that the dataset is split correctly.
    """

    def mock_choice(*args, **kwargs):
        return np.array([1, 2])

    monkeypatch.setattr(np.random, "choice", mock_choice)

    valid_dict_out, train_dict_out = ds.split_dataset(
        test_case_1["dataset_dict"], test_case_1["split"]
    )

    assert valid_dict_out.keys() == test_case_1["valid_dict_expected"].keys()
    for key in test_case_1["valid_dict_expected"].keys():
        assert (
            valid_dict_out[key] == test_case_1["valid_dict_expected"][key]
        ).all()
    assert train_dict_out.keys() == test_case_1["train_dict_expected"].keys()
    for key in test_case_1["train_dict_expected"].keys():
        assert (
            train_dict_out[key] == test_case_1["train_dict_expected"][key]
        ).all()
