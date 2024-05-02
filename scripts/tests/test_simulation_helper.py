"""
Tests for the simulation_helper.py module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

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

import logging
import multiprocessing as mp
import pathlib
import subprocess
from unittest.mock import patch

import pytest

import scripts.simulation_helper as sh


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
    sh.run_simulation_dask(command)

    # Assert that subprocess.run() was called with the correct command.
    mock_subprocess_run.assert_called_once_with(
        command, shell=True, check=True
    )

    # Check log messages
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
    sh.setup_process_pool(event, lock)

    command = "some_command"
    result = sh.run_simulation(command)
    assert result[0] == command
    assert result[1] == "Mocked subprocess output"


def test_log_simulation(caplog):
    """
    Test if the log information output is produced correct.
    """
    process_result = (pathlib.Path("some_path"), "some_output")
    caplog.set_level(logging.INFO)

    # Call the function.
    sh.log_simulation(process_result)

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
    sh.setup_process_pool(event, lock)
    assert isinstance(sh.unpaused, type(mp.Event))
    assert isinstance(sh.starting, type(mp.Lock))
