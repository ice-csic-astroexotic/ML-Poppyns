"""
    Tests for the PIC_generate_htcondor_failed module.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import argparse

import pandas as pd
import pytest

from utilities.PIC_scripts.PIC_generate_htcondor_failed import (
    generate_htcondor_failed,
)


@pytest.fixture
def args(tmp_path):
    """
    Defining the arguments needed to test the PIC_generate_htcondor_submit functions.
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            Args
    """

    return argparse.Namespace(
        output_dir_htcondor=tmp_path / "output_htcondor",
        output_dir_simulation=tmp_path / "output_simulation",
        n_sim_job=150,
        n_sim_week=100,
        dyn_data="/path/to/dyn_data",
        type_simulation="dyn",
    )


@pytest.fixture()
def mock_failed_folders_csv(tmp_path):
    """
    Create a mock `failed_folders.csv` file.
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (pathlib.Path): temporary directory where the `failed_folders.csv` file is saved.
    """
    csv_path = tmp_path / "failed_folders.csv"
    with open(csv_path, "w") as f:
        f.write("folder\n")
        f.write("000001\n")
        f.write("000002\n")
        f.write("000003\n")

    return csv_path


@pytest.fixture()
def create_folders_in_output_dir(mock_failed_folders_csv, tmp_path):
    """
    Creating the folders needed to test `generate_htcondor_failed` function.
        Args:
            mock_failed_folders_csv (Callable): Module that generates the temporary directory
                where the `failed_folders.csv` file is saved.
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (pathlib.Path): temporary directory where the mock simulations are saved.
    """
    # Read folder names from the CSV.
    df = pd.read_csv(mock_failed_folders_csv)
    folder_names = df["folder"].tolist()

    # Create the folders within output_dir_simulation.
    output_dir_simulation = tmp_path / "output_simulations"
    output_dir_simulation.mkdir()
    for folder_name in folder_names:
        (output_dir_simulation / f"{folder_name:06}").mkdir()

    return output_dir_simulation


def test_generate_htcondor_failed_dyn(
    mock_failed_folders_csv, create_folders_in_output_dir, tmp_path
):
    """
    Testing the creation of the necessary folders and files for relaunching the failed dynamical simulations created
    by the generate_htcondor_failed file.
    """
    args = argparse.Namespace(
        output_dir_simulation=str(create_folders_in_output_dir),
        dir_failed_folder_csv=str(tmp_path),
        number_sim_job=5,
        dyn_data="/path/to/dyn_data",
        type_simulation="dyn",
    )

    generate_htcondor_failed(args)

    # Check if the required directories and files are created.
    output_failed_simulations = (
        tmp_path / "failed_simulations" / "output_simulations"
    )
    htcondor_failed_submit_path = (
        tmp_path / "failed_simulations" / "htcondor_submit"
    )
    htcondor_failed_output_path = (
        tmp_path / "failed_simulations" / "htcondor_output"
    )

    assert output_failed_simulations.exists()
    assert htcondor_failed_submit_path.exists()
    assert htcondor_failed_output_path.exists()

    # Check if the submit file and wrapper file are created.
    failed_submit_path = htcondor_failed_submit_path / "job.submit"
    failed_wrapper_path = htcondor_failed_submit_path / "wrapper.sh"
    failed_arguments_path = htcondor_failed_submit_path / "arguments_.txt"

    assert failed_submit_path.exists()
    assert failed_wrapper_path.exists()
    assert failed_arguments_path.exists()


def test_generate_htcondor_failed_magrot(
    mock_failed_folders_csv, create_folders_in_output_dir, tmp_path
):
    """
    Testing the creation of the necessary folders and files for relaunching the failed magneto-rotational simulations
    created by the generate_htcondor_failed file.
    """
    args = argparse.Namespace(
        output_dir_simulation=str(create_folders_in_output_dir),
        dir_failed_folder_csv=str(tmp_path),
        number_sim_job=5,
        dyn_data="/path/to/dyn_data",
        type_simulation="magrot",
    )

    generate_htcondor_failed(args)

    # Check if the required directories and files are created.
    output_failed_simulations = (
        tmp_path / "failed_simulations" / "output_simulations"
    )
    htcondor_failed_submit_path = (
        tmp_path / "failed_simulations" / "htcondor_submit"
    )
    htcondor_failed_output_path = (
        tmp_path / "failed_simulations" / "htcondor_output"
    )

    assert output_failed_simulations.exists()
    assert htcondor_failed_submit_path.exists()
    assert htcondor_failed_output_path.exists()

    # Check if the submit file and wrapper file are created.
    failed_submit_path = htcondor_failed_submit_path / "job.submit"
    failed_wrapper_path = htcondor_failed_submit_path / "wrapper.sh"
    failed_arguments_path = htcondor_failed_submit_path / "arguments_.txt"

    assert failed_submit_path.exists()
    assert failed_wrapper_path.exists()
    assert failed_arguments_path.exists()


def test_generate_htcondor_failed_invalid_type(
    mock_failed_folders_csv, create_folders_in_output_dir, tmp_path
):
    """
    Testing the generate_htcondor_failed script when the type_simulation argument is not valid.
    """
    args = argparse.Namespace(
        output_dir_simulation=str(create_folders_in_output_dir),
        dir_failed_folder_csv=str(tmp_path),
        number_sim_job=5,
        dyn_data="/path/to/dyn_data",
        type_simulation="invalid",
    )

    with pytest.raises(ValueError) as excinfo:
        generate_htcondor_failed(args)

    assert (
        str(excinfo.value)
        == "The specified simulation type is not feasible. Choose between dyn or magrot."
    )
