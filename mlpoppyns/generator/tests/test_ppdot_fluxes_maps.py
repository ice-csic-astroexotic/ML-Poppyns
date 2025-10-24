"""
    Test for the ppdot_fluxes_maps module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import os

import numpy as np
import pytest
from pypopsyn.generator.maps.ppdot_fluxes_maps import generate_ppdot_fluxes_map


@pytest.fixture()
def test_case_1():
    data = {
        "map_name": "test_map",
        "sample_number": 1,
        "map_type": "array",
        "x_positions": np.linspace(-10, 10, 100),
        "y_positions": np.linspace(-10, 10, 100),
        "fluxes": np.random.uniform(0, 1, size=100),
        "x_resolution": 32,
        "y_resolution": 32,
        "ppdot_fluxes_maps_dictionary": {},
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


def test_generate_ppdot_fluxes_map(temp_dir, test_case_1):
    """
    Test function for generate_ppdot_fluxes_map.

    Args:
        temp_dir (str): Path to the temporary directory created by the fixture.
        test_case_1 (dict): Input args.
    """
    # Call the function to generate velocity map.
    generate_ppdot_fluxes_map(
        dataset_path=temp_dir,
        map_name=test_case_1["map_name"],
        sample_number=test_case_1["sample_number"],
        map_type=test_case_1["map_type"],
        x_positions=test_case_1["x_positions"],
        y_positions=test_case_1["y_positions"],
        fluxes=test_case_1["fluxes"],
        flux_avg_min=0.0,
        x_resolution=test_case_1["x_resolution"],
        y_resolution=test_case_1["y_resolution"],
        ppdot_fluxes_maps_dictionary=test_case_1[
            "ppdot_fluxes_maps_dictionary"
        ],
    )

    # Check if the map file is created.
    map_file = os.path.join(
        temp_dir,
        "{}_{}.npy".format(
            test_case_1["map_name"], test_case_1["sample_number"]
        ),
    )
    assert os.path.exists(map_file)

    # Check if the map file is added to the maps dictionary.
    assert (
        "input:" + test_case_1["map_name"]
        in test_case_1["ppdot_fluxes_maps_dictionary"]
    )
    assert (
        map_file
        in test_case_1["ppdot_fluxes_maps_dictionary"][
            "input:" + test_case_1["map_name"]
        ]
    )
