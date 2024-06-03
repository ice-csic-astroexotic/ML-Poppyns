"""
Test for the maps2d_generator.py module

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

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

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypopsyn.generator.maps.maps2d_generator import (
    generate_avg_weight_map,
    generate_avg_weight_matrix,
    generate_density_map,
    generate_density_matrix,
)


@pytest.fixture()
def test_case_1():
    data = {
        "x": np.random.uniform(0, 100, size=1000),
        "y": np.random.uniform(0, 100, size=1000),
        "w": np.random.uniform(0, 1, size=1000),
        "x_range": (0, 100),
        "y_range": (0, 100),
    }

    return data


@pytest.fixture
def temp_output_path():
    """
    Fixture to create a temporary directory for test outputs.

    Yields:
        (str): Temporary directory path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_generate_density_map(temp_output_path, test_case_1):
    """
    Test case for generate_density_map function.

    Args:
        temp_output_path (str): Temporary directory path for test outputs.
    """
    # Generate some sample data.
    x = test_case_1["x"]
    y = test_case_1["y"]
    filename = os.path.join(temp_output_path, "density_map.png")

    # Check if the .png map exists.
    generate_density_map(
        x, test_case_1["x_range"], y, test_case_1["y_range"], filename
    )
    assert os.path.exists(filename)


def test_generate_avg_weight_map(temp_output_path, test_case_1):
    """
    Test case for generate_avg_weight_map function.

    Args:
        temp_output_path (str): Temporary directory path for test outputs.
    """
    # Generate some sample data.
    x = test_case_1["x"]
    y = test_case_1["y"]
    w = test_case_1["w"]
    filename = os.path.join(temp_output_path, "avg_weight_map.png")

    # Check if the .png map exists.
    generate_avg_weight_map(
        x, test_case_1["x_range"], y, test_case_1["y_range"], w, filename
    )
    assert os.path.exists(filename)


def test_generate_density_matrix(temp_output_path, test_case_1):
    """
    Test case for generate_density_matrix function.

    Args:
        temp_output_path (str): Temporary directory path for test outputs.
    """
    # Generate some sample data
    x = test_case_1["x"]
    y = test_case_1["y"]
    filename = os.path.join(temp_output_path, "density_matrix.npy")

    # Check if the .npy matrix exists.
    generate_density_matrix(
        x, test_case_1["x_range"], y, test_case_1["y_range"], filename
    )
    assert os.path.exists(filename)


def test_generate_avg_weight_matrix(temp_output_path, test_case_1):
    """
    Test case for generate_avg_weight_matrix function.

    Args:
        temp_output_path (str): Temporary directory path for test outputs.
    """
    # Generate some sample data
    x = test_case_1["x"]
    y = test_case_1["y"]
    w = test_case_1["w"]
    filename = os.path.join(temp_output_path, "avg_weight_matrix.npy")

    # Check if the .npy matrix exists.
    generate_avg_weight_matrix(
        x, test_case_1["x_range"], y, test_case_1["y_range"], w, filename
    )
    assert os.path.exists(filename)
