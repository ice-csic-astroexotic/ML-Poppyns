"""
    Test for the ppdot_maps module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import os

import numpy as np
import pytest

from pypopsyn.generator.maps.ppdot_maps import generate_ppdot_map


@pytest.fixture()
def test_case_1():
    data = {
        "map_name": "test_map",
        "sample_number": 1,
        "map_type": "array",
        "periods": np.linspace(0.01, 10, 100),
        "period_derivatives": np.linspace(1e-19, 1.0e-10, 100),
        "p_resolution": 32,
        "pdot_resolution": 32,
        "ppdot_maps_dictionary": {},
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


def test_generate_ppdot_map(temp_dir, test_case_1):
    """
    Test function for generate_ppdot_map.

    Args:
        temp_dir (str): Path to the temporary directory created by the fixture.
        test_case_1 (dict): input args.
    """
    # Call the function to generate position map.
    generate_ppdot_map(
        dataset_path=temp_dir,
        map_name=test_case_1["map_name"],
        sample_number=test_case_1["sample_number"],
        map_type=test_case_1["map_type"],
        periods=test_case_1["periods"],
        period_derivatives=test_case_1["period_derivatives"],
        p_resolution=test_case_1["p_resolution"],
        pdot_resolution=test_case_1["pdot_resolution"],
        ppdot_maps_dictionary=test_case_1["ppdot_maps_dictionary"],
    )

    # Check if the map file is created.
    map_file = os.path.join(
        temp_dir,
        "{}_{}.npy".format(
            test_case_1["map_name"], test_case_1["sample_number"]
        ),
    )
    assert os.path.exists(map_file)

    # Check if the map file is added to the ppdot maps dictionary.
    assert (
        "input:" + test_case_1["map_name"]
        in test_case_1["ppdot_maps_dictionary"]
    )
    assert (
        map_file
        in test_case_1["ppdot_maps_dictionary"][
            "input:" + test_case_1["map_name"]
        ]
    )
