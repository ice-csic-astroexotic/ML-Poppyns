"""
    Tests for the PIC_generate_htcondor_submit.py module.

    Authors:

        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import argparse
import pathlib

import pytest

from utilities.PIC_scripts.PIC_generate_htcondor_submit import (
    generate_job_submit,
    generate_wrapper,
    submit_generator,
)


@pytest.fixture
def args(tmp_path):
    """
    Defining the arguments needed to test the PIC_generate_htcondor_submit functions when type_simulation argument is
    set to "dyn".
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (Args): Arguments needed to test the PIC_generate_htcondor_submit functions.
    """
    return argparse.Namespace(
        output_dir_htcondor=tmp_path / "output_htcondor",
        output_dir_simulation=tmp_path / "output_simulation",
        n_sim_job=150,
        n_sim_week=100,
        dyn_data="/path/to/dyn_data",
        type_simulation="dyn",
    )


@pytest.fixture
def args_magrot(tmp_path):
    """
    Defining the arguments needed to test the PIC_generate_htcondor_submit functions when type_simulation argument is
    set to "magrot".
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (Args): Arguments needed to test the PIC_generate_htcondor_submit functions.
    """
    return argparse.Namespace(
        output_dir_htcondor=tmp_path / "output_htcondor",
        output_dir_simulation=tmp_path / "output_simulation",
        n_sim_job=150,
        n_sim_week=100,
        dyn_data="/path/to/dyn_data",
        type_simulation="magrot",
    )


@pytest.fixture
def args_invalid(tmp_path):
    """
    Defining the arguments needed to test the PIC_generate_htcondor_submit functions when type_simulation argument is
    invalid.
        Args:
            tmp_path (pathlib.Path): Temporary directory created automatically by pytest.
        Returns:
            (Args): Arguments needed to test the PIC_generate_htcondor_submit functions.
    """
    return argparse.Namespace(
        output_dir_htcondor=tmp_path / "output_htcondor",
        output_dir_simulation=tmp_path / "output_simulation",
        n_sim_job=150,
        n_sim_week=100,
        dyn_data="/path/to/dyn_data",
        type_simulation="invalid",
    )


def create_simulation_arguments_file(output_dir_simulation):
    """
    Create the `simulation_arguments.txt` file with mock data.
    """
    with open(output_dir_simulation / "simulation_arguments.txt", "w") as file:
        for i in range(1, 5):
            folder_name = f"{i:06}"
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
    Testing the generation of the `wrapper.sh` file by the generate_wrapper function when the type_simulation
    argument is set to "dyn".
    """
    wrapper_path = pathlib.Path(tmp_path) / "wrapper.sh"
    generate_wrapper(
        args.type_simulation, pathlib.Path(args.dyn_data), wrapper_path
    )

    assert wrapper_path.exists()


def test_generate_wrapper_magrot(args_magrot, tmp_path):
    """
    Testing the generation of the `wrapper.sh` file by the generate_wrapper function when the type_simulation
    argument is set to "magrot".
    """
    wrapper_path = pathlib.Path(tmp_path) / "wrapper.sh"
    generate_wrapper(
        args_magrot.type_simulation,
        pathlib.Path(args_magrot.dyn_data),
        wrapper_path,
    )

    assert wrapper_path.exists()


def test_generate_wrapper_invalid(args_invalid, tmp_path):
    """
    Testing the generation of the `wrapper.sh` file by the generate_wrapper function when the type_simulation
    argument is invalid.
    """
    wrapper_path = pathlib.Path(tmp_path) / "wrapper.sh"

    with pytest.raises(ValueError) as excinfo:
        generate_wrapper(
            args_invalid.type_simulation,
            pathlib.Path(args_invalid.dyn_data),
            wrapper_path,
        )

    assert (
        str(excinfo.value)
        == "The specified simulation type is not feasible, choose between dyn or magrot."
    )


def test_submit_generator(args, tmp_path):
    """
    Testing the generation of the folders needed to launch the simulations in HTCondor by the submit_generator function.
    """

    output_dir_simulation = args.output_dir_simulation
    output_dir_htcondor = args.output_dir_htcondor

    output_dir_simulation.mkdir(parents=True)

    # Create the simulation arguments file.
    create_simulation_arguments_file(output_dir_simulation)

    submit_generator(args)

    assert output_dir_htcondor.exists()
    assert (output_dir_htcondor / "week-0").exists()
    assert (output_dir_htcondor / "week-0" / "job.submit").exists()
    assert (output_dir_htcondor / "week-0" / "arguments_week.txt").exists()
