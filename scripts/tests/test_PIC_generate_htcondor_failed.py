"""
Tests for the PIC_generate_hctondor_failed.py module.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)

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

import argparse

import pandas as pd
import pytest

from scripts.PIC_generate_htcondor_failed import generate_htcondor_failed


@pytest.fixture()
def mock_failed_folders_csv(tmp_path):
    # Create a mock failed_folders.csv file
    csv_path = tmp_path / "failed_folders.csv"
    with open(csv_path, "w") as f:
        f.write("folder\n")
        f.write("000001\n")
        f.write("000002\n")
        f.write("000003\n")

    return csv_path


@pytest.fixture()
def create_folders_in_output_dir(mock_failed_folders_csv, tmp_path):
    # Read folder names from the CSV
    df = pd.read_csv(mock_failed_folders_csv)
    folder_names = df["folder"].tolist()

    # Create the folders within output_dir_simulation
    output_dir_simulation = tmp_path / "output_simulations"
    output_dir_simulation.mkdir()
    for folder_name in folder_names:
        (output_dir_simulation / f"{folder_name:06}").mkdir()

    return output_dir_simulation


def test_generate_htcondor_failed(
    mock_failed_folders_csv, create_folders_in_output_dir, tmp_path
):

    args = argparse.Namespace(
        output_dir_simulation=str(create_folders_in_output_dir),
        dir_failed_folder_csv=str(tmp_path),
        number_sim_job=5,
        dyn_data="/path/to/dyn_data",
        type_simulation="dyn",
    )

    generate_htcondor_failed(args)

    # Check if the required directories and files are created
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

    # Check if the submit file and wrapper file are created
    failed_submit_path = htcondor_failed_submit_path / "job.submit"
    failed_wrapper_path = htcondor_failed_submit_path / "wrapper.sh"
    failed_arguments_path = htcondor_failed_submit_path / "arguments_.txt"

    assert failed_submit_path.exists()
    assert failed_wrapper_path.exists()
    assert failed_arguments_path.exists()
