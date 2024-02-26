"""
Tests for the parameter_set_generator.py module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

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

from pypopsyn.simulator.configuration import cfg
from scripts import parameter_set_generator as psg


@pytest.fixture()
def test_case_1():
    data = {
        "parameter_name": "sigma_k",
        "args_dict": {
            "output_dir": None,
            "sampling_type": "random",
            "sampling_size": 5,
            "sigma_k": None,
            "vk_c": None,
            "h_c": [0.1, 1.0],
            "P_initial_mean": None,
            "P_initial_sigma": None,
            "P_initial_log10_mean": None,
            "P_initial_log10_sigma": None,
            "B_initial_log10_mean": None,
            "B_initial_log10_sigma": None,
            "a_late": None,
        },
        "var_names_expected": ["h_c"],
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "parameter_name": "sigma_k",
        "args_dict": {
            "output_dir": None,
            "sampling_type": "grid",
            "sigma_k": None,
            "vk_c": None,
            "h_c": [0.1, 1.0, 5],
            "P_initial_mean": None,
            "P_initial_sigma": None,
            "P_initial_log10_mean": None,
            "P_initial_log10_sigma": None,
            "B_initial_log10_mean": None,
            "B_initial_log10_sigma": None,
            "a_late": None,
        },
        "h_c_expected": np.array([0.1, 0.325, 0.55, 0.775, 1.0]),
        "var_names_expected": [
            "h_c",
        ],
        "var_expanded_ranges_expected": [
            [0.1, 0.325, 0.55, 0.775, 1.0],
        ],
    }

    return data


def test_expand_parameter_random(test_case_1):
    """
    Testing if a parameter is correctly expanded in random mode.
    """
    expanded_hc = psg.expand_parameter(
        test_case_1["args_dict"], "h_c", test_case_1["args_dict"]["h_c"]
    )

    assert len(expanded_hc) == test_case_1["args_dict"]["sampling_size"]


def test_check_expand_args_random(test_case_1):
    """
    Testing if the arguments are correctly checked and expanded in random mode.
    """
    var_names_out, var_expanded_ranges_out = psg.check_expand_args(
        test_case_1["args_dict"]
    )

    assert var_names_out == test_case_1["var_names_expected"]

    assert np.shape(var_expanded_ranges_out) == (
        len(test_case_1["var_names_expected"]),
        test_case_1["args_dict"]["sampling_size"],
    )


def test_expand_parameter_grid(test_case_2):
    """
    Testing if a parameter is correctly expanded in grid mode.
    """
    expanded_hc = psg.expand_parameter(
        test_case_2["args_dict"], "h_c", test_case_2["args_dict"]["h_c"]
    )

    assert (expanded_hc == test_case_2["h_c_expected"]).all()


def test_check_expand_args_grid(test_case_2):
    """
    Testing if the arguments are correctly checked and expanded in grid mode.
    """
    var_names_out, var_expanded_ranges_out = psg.check_expand_args(
        test_case_2["args_dict"]
    )

    assert var_names_out == test_case_2["var_names_expected"]

    assert (
        var_expanded_ranges_out == test_case_2["var_expanded_ranges_expected"]
    )
