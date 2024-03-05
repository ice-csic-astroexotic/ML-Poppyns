"""
Test for the PIC_manage_failed_simulation.py module.

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
import json

import pytest

from scripts.PIC_manage_failed_simulation import manage_failed_simulations


# Define fixtures to set up a sample environment for testing.
@pytest.fixture
def failed_simulation_dir(tmp_path):
    """
    Creating the structure and files inside the `failed_simulation` directory needed to test the
    manage_failed_simulations function.
    """
    # Create a temporary failed simulation directory.
    failed_dir = tmp_path / "failed_simulation"
    failed_dir.mkdir()
    # Create a temporary output_simulations directory inside the failed_simulation directory.
    output_failed_dir = failed_dir / "output_simulations"
    output_failed_dir.mkdir()

    # Create sample folders and files inside failed_simulation directory.
    for i in range(1, 4):
        sample_folder = output_failed_dir / f"{i:06}"
        sample_folder.mkdir()
        # Create mock override.json file inside each folder.
        sample_file = sample_folder / "override.json"
        sample_content = {"key": f"value {i}"}
        with open(sample_file, "w") as json_file:
            json.dump(sample_content, json_file)

    return failed_dir


@pytest.fixture
def simulation_dir(tmp_path):
    """
    Creating the structure inside the `output_simulations` directory needed to test the `manage_failed_simulations`
    function.
    """
    # Create a temporary simulation directory.
    sim_dir = tmp_path / "output_simulations"
    sim_dir.mkdir()

    # Create sample folders and files inside simulation directory.
    for i in range(1, 4):
        sample_folder = sim_dir / f"{i:06}"
        sample_folder.mkdir()

    return sim_dir


def test_manage_failed_simulations(failed_simulation_dir, simulation_dir):
    """
    Testing that the files within the `failed_simulation_dir` directory are copied back to the original folders in the
     `simulation_dir` directory.
    """
    args = argparse.Namespace(
        failed_simulation_dir=failed_simulation_dir,
        simulation_dir=simulation_dir,
    )
    manage_failed_simulations(args)

    # Assert that the files from failed_simulation_dir are copied back to simulation_dir.
    for i in range(1, 4):

        assert not (simulation_dir / f"{i:06}" / "override.json").exists()
