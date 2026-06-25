"""
    Test for the position_maps module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import os

import numpy as np
import pytest

import mlpoppyns.generator.maps.density_map_generator as dmap


@pytest.fixture()
def test_case_1():
    data = {
        "map_name": "test_map",
        "sample_number": 1,
        "map_type": "array",
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
        "x_resolution": 32,
        "y_resolution": 32,
        "x_log_scale": False,
        "y_log_scale": False,
        "x_limits": (-10, 10),
        "y_limits": (-10, 10),
        "position_maps_dictionary": {},
    }

    return data


# Fixture to create a temporary directory for testing
@pytest.fixture
def temp_dir(tmpdir):
    """
    Fixture to create a temporary directory for testing.

    Args:
        (tmpdir): Pytest's built-in fixture to create temporary directories.

    Yields:
        (str): The path to the temporary directory.
    """
    yield str(tmpdir)
    tmpdir.remove()


def test_generate_density_map(temp_dir, test_case_1):
    """
    Test function for generate_position_map.

    Args:
        temp_dir (str): Path to the temporary directory created by the fixture.
        test_case_1 (dict): input args.
    """
    # Call the function to generate position map.
    dmap.generate_density_map(
        dataset_path=temp_dir,
        map_name=test_case_1["map_name"],
        sample_number=test_case_1["sample_number"],
        map_type=test_case_1["map_type"],
        x=test_case_1["x"],
        y=test_case_1["y"],
        x_resolution=test_case_1["x_resolution"],
        y_resolution=test_case_1["y_resolution"],
        x_log_scale=test_case_1["x_log_scale"],
        y_log_scale=test_case_1["y_log_scale"],
        maps_dictionary=test_case_1["position_maps_dictionary"],
        x_limits=test_case_1["x_limits"],
        y_limits=test_case_1["y_limits"],
    )

    # Check if the map file is created.
    map_file = os.path.join(
        temp_dir,
        "{}_{}.npy".format(
            test_case_1["map_name"], test_case_1["sample_number"]
        ),
    )
    assert os.path.exists(map_file)

    # Check if the map file is added to the position maps dictionary.
    assert (
        "input:" + test_case_1["map_name"]
        in test_case_1["position_maps_dictionary"]
    )
    assert (
        map_file
        in test_case_1["position_maps_dictionary"][
            "input:" + test_case_1["map_name"]
        ]
    )
