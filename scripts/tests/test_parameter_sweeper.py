"""
Tests for the parameter_sweeper.py module.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2024

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
import argparse
import os

import pytest

import scripts.parameter_sweeper as param_sweeper


@pytest.fixture
def args_random(tmp_path):
    """
    Defining the arguments needed to test the parameter_sweeper functions when the `sampling_type` argument is set to
    "random".
    Args:
        tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
    Returns:
        Args
    """
    return argparse.Namespace(
        output_dir=tmp_path,
        sampling_type="random",
        sampling_size=10,
        kick_model="km_maxwell",
        sigma_k=[20, 790],
        vk_c=None,
        h_c=None,
        spin_period_model="log-normal",
        P_initial_mean=None,
        P_initial_sigma=None,
        P_initial_log10_mean=[-1, 1],
        P_initial_log10_sigma=None,
        B_initial_log10_mean=None,
        B_initial_log10_sigma=None,
        a_late=None,
    )


@pytest.fixture
def args_grid(tmp_path):
    """
    Defining the arguments needed to test the parameter_sweeper functions when the `sampling_type` argument is set to
    "grid".
    Args:
        tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
    Returns:
        Args
    """
    return argparse.Namespace(
        output_dir=tmp_path,
        sampling_type="grid",
        kick_model="km_maxwell",
        sigma_k=[20, 790, 10],
        vk_c=None,
        h_c=None,
        spin_period_model="log-normal",
        P_initial_mean=None,
        P_initial_sigma=None,
        P_initial_log10_mean=[-1, 1, 10],
        P_initial_log10_sigma=None,
        B_initial_log10_mean=None,
        B_initial_log10_sigma=None,
        a_late=None,
    )


@pytest.fixture
def args_invalid(tmp_path):
    """
    Defining the arguments needed to test the parameter_sweeper functions when the `sampling_type` argument is
    invalid.
    Args:
        tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
    Returns:
        Args
    """
    return argparse.Namespace(
        output_dir=tmp_path,
        sampling_type="invalid",
        kick_model="km_maxwell",
        sigma_k=[20, 790, 10],
        vk_c=None,
        h_c=None,
        spin_period_model="log-normal",
        P_initial_mean=None,
        P_initial_sigma=None,
        P_initial_log10_mean=[-1, 1, 10],
        P_initial_log10_sigma=None,
        B_initial_log10_mean=None,
        B_initial_log10_sigma=None,
        a_late=None,
    )


def test_main_random(args_random, tmp_path):
    """
    Testing the main function of the parameter_sweeper.py script when the `sampling_type` argument is set to "random".
     For this, we check if the folders and `simulation_arguments.txt` file are created within the `args.output_dir`
     folder. We also check if the `override.json` file is created inside each of the folders.
    """

    param_sweeper.main(args_random)

    # Check if the folders are created.
    assert len(os.listdir(args_random.output_dir)) == 11

    # Check if simulation_arguments.txt file exists.
    simulation_arguments_path = os.path.join(
        args_random.output_dir, "simulation_arguments.txt"
    )
    assert os.path.isfile(simulation_arguments_path)

    # Check if override.json file exists inside each folder.
    for folder in os.listdir(args_random.output_dir):
        folder_path = os.path.join(args_random.output_dir, folder)
        if os.path.isdir(folder_path):
            override_json_path = os.path.join(folder_path, "override.json")
            assert os.path.isfile(override_json_path)


def test_main_grid(args_grid, tmp_path):
    """
    Testing the main function of the parameter_sweeper.py script when the `sampling_type` argument is set to "grid".
    For this, we check if the folders and `simulation_arguments.txt` file are created within the `args.output_dir`
    folder. We also check if the `override.json` file is created inside each of the folders.
    """

    param_sweeper.main(args_grid)

    # Check if the folders are created.
    assert len(os.listdir(args_grid.output_dir)) == 101

    # Check if simulation_arguments.txt file exists.
    simulation_arguments_path = os.path.join(
        args_grid.output_dir, "simulation_arguments.txt"
    )
    assert os.path.isfile(simulation_arguments_path)

    # Check if override.json file exists inside each folder.
    for folder in os.listdir(args_grid.output_dir):
        folder_path = os.path.join(args_grid.output_dir, folder)
        if os.path.isdir(folder_path):
            override_json_path = os.path.join(folder_path, "override.json")
            assert os.path.isfile(override_json_path)


def test_main_invalid(args_invalid, tmp_path):
    """
    Testing the main function of the parameter_sweeper.py script when the `sampling_type` argument is invalid.
    """

    with pytest.raises(ValueError) as excinfo:
        param_sweeper.main(args_invalid)

    assert (
        str(excinfo.value)
        == "The value invalid is not feasible for parameter sampling_type"
    )
