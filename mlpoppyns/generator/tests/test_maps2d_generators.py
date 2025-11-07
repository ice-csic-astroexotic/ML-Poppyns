"""
    Test for the maps2d_generator module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlpoppyns.generator.maps.maps2d_generators import (
    generate_avg_weight_image,
    generate_avg_weight_matrix,
    generate_density_image,
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
    generate_density_image(
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
    generate_avg_weight_image(
        x, test_case_1["x_range"], y, test_case_1["y_range"], w, 0.0, filename
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
        x, test_case_1["x_range"], y, test_case_1["y_range"], w, 0.0, filename
    )
    assert os.path.exists(filename)
