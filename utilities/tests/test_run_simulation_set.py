"""
    Tests for the run_simulation_set module.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
from unittest import mock

import pytest

import utilities.experiment_helpers.run_simulation_set as rss


@pytest.fixture
def mock_subprocess_check_output(monkeypatch):
    def mock_check_output(command, stderr, shell):
        return b"Mocked subprocess output"

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)


class MockArgs:
    def __init__(self, save_dir, parameter_override, dyn_data):
        self.save_dir = save_dir
        self.parameter_override = parameter_override
        self.dyn_data = dyn_data


def test_run_simulation_dask(caplog):
    """
    Test the run_simulation_dask method.
    """
    # Define mock inputs.
    dyn_data_path = "mock_dyn_data_path"

    args = MockArgs(
        "mock_output_dir", "mock_parameter_override.json", dyn_data_path
    )
    simulator_type = "simulate_population_magrot_det"
    simulation_output_path = "mock_simulation_output_path"
    simulation_override_json = {"param1": "value1", "param2": "value2"}

    caplog.set_level(logging.INFO)

    # Mock the necessary functions and methods used in run_simulation_dask.
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "shutil.copytree"
    ), mock.patch("pathlib.Path.mkdir"), mock.patch("json.dump"), mock.patch(
        "shutil.rmtree"
    ), mock.patch(
        "mlpoppyns.simulator.simulate_population_magrot_det.simulate_population"
    ):
        try:
            # Call the function being tested.
            rss.run_simulation_dask(
                args,
                simulator_type,
                simulation_output_path,
                simulation_override_json,
                dyn_data_path,
            )
            # Assert that the log messages are present.
            assert (
                "Copied output folder back to original location" in caplog.text
            )
            assert "Simulation finished" in caplog.text
        finally:
            # Ensure the file is deleted after the test.
            if os.path.exists("mock_parameter_override.json"):
                os.remove("mock_parameter_override.json")


def test_run_simulation(mock_subprocess_check_output):
    """
    Test the run_simulation method.
    """
    # Create dummy event and lock.
    event = mp.Event()
    lock = mp.Lock()

    # Set up the process pool.
    rss.setup_process_pool(event, lock)

    command = "some_command"
    result = rss.run_simulation(command)
    assert result[0] == command
    assert result[1] == "Mocked subprocess output"


def test_log_simulation(caplog):
    """
    Test if the log information output is produced correct.
    """
    process_result = (pathlib.Path("some_path"), "some_output")
    caplog.set_level(logging.INFO)

    # Call the function.
    rss.log_simulation(process_result)

    # Print the captured log records.
    print("Captured log records:")
    for record in caplog.records:
        print(record.levelname, record.message)

    # Verify that logs are captured.
    assert "Ran simulation" in caplog.text
    assert "some_output" in caplog.text


def test_setup_process_pool():
    """
    Test the setup_process_pool method.
    """
    event = mp.Event
    lock = mp.Lock
    rss.setup_process_pool(event, lock)
    assert isinstance(rss.unpaused, type(mp.Event))
    assert isinstance(rss.starting, type(mp.Lock))
