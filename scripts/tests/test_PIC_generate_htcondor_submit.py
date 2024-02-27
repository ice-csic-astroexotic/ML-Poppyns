"""
Tests for the PIC_generate_hctondor_submit.py module.

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
import pathlib

import pytest

from scripts.PIC_generate_htcondor_submit import (
    generate_job_submit,
    generate_wrapper,
    submit_generator,
)


@pytest.fixture
def args(tmp_path):
    """
    Defining the arguments needed to test the PIC_generate_hctondor_submit functions.
    """
    return argparse.Namespace(
        output_dir_htcondor=tmp_path / "output_htcondor",
        output_dir_simulation=tmp_path / "output_simulation",
        n_sim_job=150,
        n_sim_week=100,
        dyn_data="/path/to/dyn_data",
        type_simulation="dyn",
    )


def create_simulation_arguments_file(output_dir_simulation):
    """
    Create the `simulation_arguments.txt` file with mock data.
    """
    with open(output_dir_simulation / "simulation_arguments.txt", "w") as file:
        for i in range(1, 5):  # Change the range as needed
            folder_name = f"{i:06}"  # Format the number with leading zeros
            file.write(
                f"{output_dir_simulation}/{folder_name} {output_dir_simulation}/{folder_name}/override.json\n"
            )


def test_generate_job_submit(args, tmp_path):
    """
    Testing the generation of the `job.submit` file by the generate_job_submit function.
    """
    output_dir = pathlib.Path(tmp_path) / args.output_dir_htcondor
    output_dir.mkdir()

    arguments_path = output_dir / "arguments_week.txt"
    generate_job_submit(output_dir, arguments_path)
    submit_path = output_dir / "job.submit"
    assert submit_path.exists()


def test_generate_wrapper(args, tmp_path):
    """
    Testing the generation of the `wrapper.sh` file by the generate_wrapper function.
    """
    wrapper_path = pathlib.Path(tmp_path) / "wrapper.sh"
    generate_wrapper(
        args.type_simulation, pathlib.Path(args.dyn_data), wrapper_path
    )

    assert wrapper_path.exists()


def test_submit_generator(args, tmp_path):
    """
    Testing the generation of the folders needed to launch the simulations in HTCondor by the submit_generator function.
    """

    output_dir_simulation = args.output_dir_simulation
    output_dir_htcondor = args.output_dir_htcondor

    output_dir_simulation.mkdir(parents=True)

    # Create the simulation arguments file
    create_simulation_arguments_file(output_dir_simulation)

    submit_generator(args)

    assert output_dir_htcondor.exists()
    assert (output_dir_htcondor / "week-0").exists()
    assert (output_dir_htcondor / "week-0" / "job.submit").exists()
    assert (output_dir_htcondor / "week-0" / "arguments_week.txt").exists()
