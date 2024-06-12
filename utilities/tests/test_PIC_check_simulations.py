"""
    Tests for the PIC_check_simulations.py module.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import argparse
import os
import pathlib

import pandas as pd
import pytest

from utilities.PIC_scripts.PIC_check_simulations import check_simulations


@pytest.fixture()
def mock_simulation_folders(tmp_path):
    """
    Create mock simulation folders needed for testing.
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (pathlib.Path, pathlib.Path): The temporary directories for saving the simulations and failed simulations.
    """

    simulation_folders = [
        "000001",
        "000002",
        "000003",
        "000004",
        "000005",
    ]
    failed_simulation_folders = ["000002", "000004"]

    for folder in simulation_folders:
        folder_path = tmp_path / folder
        os.makedirs(folder_path)
        # Create files in each folder.
        for i in range(8):
            open(os.path.join(folder_path, f"file_{i}.txt"), "w").close()

    # Mark failed simulation folders.
    for folder in failed_simulation_folders:
        folder_path = tmp_path / folder
        # Remove files to simulate failure.
        for i in range(3):
            os.remove(os.path.join(folder_path, f"file_{i}.txt"))

    return simulation_folders, failed_simulation_folders


def test_check_simulations(mock_simulation_folders, tmp_path):
    """
    Testing the `check_simulations` function to ensure it creates the proper folder structure for failed simulations.
    """
    simulation_folders, failed_simulation_folders = mock_simulation_folders

    args = argparse.Namespace(
        output_dir_simulation=str(tmp_path),
        dir_failed_folder_csv=str(tmp_path),
    )
    check_simulations(args)

    failed_folder_csv_path = pathlib.Path().joinpath(
        tmp_path, "failed_folders.csv"
    )
    # Check if `failed_folders.csv` is created.
    assert os.path.exists(failed_folder_csv_path)

    df = pd.read_csv(failed_folder_csv_path)

    list_failed_folder = [f"{folder:06}" for folder in df["folder"]]

    # Check if the folders are correctly created for the failed simulations.
    assert len(df) == len(failed_simulation_folders)
    assert set(list_failed_folder) == set(failed_simulation_folders)
