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

import logging

import numpy as np
import pytest

from pypopsyn.simulator.configuration import cfg
from scripts import parameter_set_generator as psg


@pytest.fixture()
def test_case_1():
    data = {
        "args_dict": {
            "sigma_k": [10, 500],
            "vk_c": None,
            "h_c": None,
            "P_initial_mean": None,
            "P_initial_sigma": None,
            "P_initial_log10_mean": [-2, -1],
            "P_initial_log10_sigma": None,
            "B_initial_log10_mean": None,
            "B_initial_log10_sigma": None,
            "a_late": None,
        },
        "expected_output": "The provided parameters are compatible with the configuration file.",
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "args_dict": {
            "sigma_k": None,
            "vk_c": [10, 500],
            "h_c": None,
            "P_initial_mean": None,
            "P_initial_sigma": None,
            "P_initial_log10_mean": None,
            "P_initial_log10_sigma": None,
            "B_initial_log10_mean": None,
            "B_initial_log10_sigma": None,
            "a_late": None,
        },
        "expected_output": "The provided kick-velocity distribution parameter is not compatible with "
        "the model {} in the configuration file.".format(cfg["kick_model"]),
    }

    return data


@pytest.fixture()
def test_case_3():
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
def test_case_4():
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


def test_check_parameter_compatibility_compatible(test_case_1, caplog):
    """
    Testing if the parsed parameters are compatible with the models in the configuration file.
    """
    caplog.set_level(logging.INFO)
    psg.check_parameter_compatibility(test_case_1["args_dict"])

    assert test_case_1["expected_output"] in caplog.text


def test_check_parameter_compatibility_not_compatible(test_case_2):
    """
    Testing if the parsed parameters are not compatible with the models in the configuration file.
    """

    with pytest.raises(ValueError) as excinfo:
        psg.check_parameter_compatibility(test_case_2["args_dict"])

    # Assert that the raised exception has the expected error message.
    assert str(excinfo.value) == test_case_2["expected_output"]


def test_expand_parameter_random(test_case_3):
    """
    Testing if a parameter is correctly expanded in random mode.
    """
    expanded_hc = psg.expand_parameter(
        test_case_3["args_dict"], "h_c", test_case_3["args_dict"]["h_c"]
    )

    assert len(expanded_hc) == test_case_3["args_dict"]["sampling_size"]


def test_check_expand_args_random(test_case_3):
    """
    Testing if the arguments are correctly checked and expanded in random mode.
    """
    var_names_out, var_expanded_ranges_out = psg.check_expand_args(
        test_case_3["args_dict"]
    )

    assert var_names_out == test_case_3["var_names_expected"]

    assert np.shape(var_expanded_ranges_out) == (
        len(test_case_3["var_names_expected"]),
        test_case_3["args_dict"]["sampling_size"],
    )


def test_expand_parameter_grid(test_case_4):
    """
    Testing if a parameter is correctly expanded in grid mode.
    """
    expanded_hc = psg.expand_parameter(
        test_case_4["args_dict"], "h_c", test_case_4["args_dict"]["h_c"]
    )

    assert (expanded_hc == test_case_4["h_c_expected"]).all()


def test_check_expand_args_grid(test_case_4):
    """
    Testing if the arguments are correctly checked and expanded in grid mode.
    """
    var_names_out, var_expanded_ranges_out = psg.check_expand_args(
        test_case_4["args_dict"]
    )

    assert var_names_out == test_case_4["var_names_expected"]

    assert (
        var_expanded_ranges_out == test_case_4["var_expanded_ranges_expected"]
    )
