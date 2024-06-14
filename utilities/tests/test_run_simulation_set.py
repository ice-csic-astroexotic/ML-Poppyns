"""
    Tests for the run_simulation_set module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import logging
import multiprocessing as mp
import pathlib
import subprocess
from unittest.mock import patch

import pytest

import utilities.simulation_helper.run_simulation_set as rss


@pytest.fixture
def mock_subprocess_check_output(monkeypatch):
    def mock_check_output(command, stderr, shell):
        return b"Mocked subprocess output"

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)


@pytest.fixture
def mock_subprocess_run():
    """
    Mock subprocess.run() function.
    """
    with patch("subprocess.run") as mock_run:
        yield mock_run


def test_run_simulation_dask(mock_subprocess_run, caplog):
    """
    Test the run_simulation_dask method.
    """
    command = "some_command"
    caplog.set_level(logging.INFO)
    # Call the function being tested.
    rss.run_simulation_dask(command)

    # Assert that subprocess.run() was called with the correct command.
    mock_subprocess_run.assert_called_once_with(
        command, shell=True, check=True
    )

    # Check log messages.
    assert "Launching simulation" in caplog.text
    assert "Simulation finished" in caplog.text


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
