"""
    Test for the timewith module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import json
import os
import time

import pytest

from utilities.benchmark.timewith import TimeWith


@pytest.fixture
def temp_files():
    """
    Fixture for creating temporary log and JSON files for testing.

    Yields:
        (Tuple): A tuple containing the filenames for the temporary log and JSON files.
    """
    log_filename = "profile.log"
    json_filename = "profile.json"
    yield log_filename, json_filename
    if os.path.exists(log_filename):
        os.remove(log_filename)
    if os.path.exists(json_filename):
        os.remove(json_filename)


def test_initialization(temp_files):
    """
    Test case to verify the initialization of the TimeWith object.

    Args:
        temp_files (fixture): Fixture providing temporary log and JSON filenames.
    """
    log_filename, json_filename = temp_files

    # Test default initialization
    tw = TimeWith(name="test_context")
    assert tw.name == "test_context"
    assert isinstance(tw.start, float)
    assert isinstance(tw.last, float)
    assert tw.log_filename == log_filename
    assert tw.json_filename == json_filename
    assert tw.show is True

    # Check if JSON file is created
    assert os.path.isfile(json_filename)

    with open(json_filename, "r") as f:
        data = json.load(f)
    assert "test_context" in data
    assert data["test_context"]["time"] == 0.0


def test_elapsed(temp_files):
    """
    Test case to verify the elapsed method of TimeWith class.

    Args:
        temp_files (fixture): Fixture providing temporary log and JSON filenames.
    """
    tw = TimeWith(name="test_context_elapsed")
    time.sleep(0.1)

    # Asserts that the elapsed method returns the correct cumulative and total elapsed times.
    cumulative, total = tw.elapsed()
    assert cumulative >= 0.1
    assert total >= 0.1

    time.sleep(0.1)
    new_cumulative, new_total = tw.elapsed()
    assert new_cumulative >= 0.2
    assert new_total >= 0.1


def test_checkpoint(temp_files):
    """
    Test case to verify the checkpoint method of TimeWith class.

    Args:
        temp_files (fixture): Fixture providing temporary log and JSON filenames.
    """
    log_filename, json_filename = temp_files

    tw = TimeWith(
        name="test_context_checkpoint",
        log_filename=log_filename,
        json_filename=json_filename,
    )
    time.sleep(0.1)
    tw.checkpoint("checkpoint1")

    # Asserts that checkpoint method logs the correct information to the specified log and JSON files.
    with open(log_filename, "r") as f:
        logs = f.readlines()
    assert "checkpoint1" in logs[-1]

    with open(json_filename, "r") as f:
        data = json.load(f)
    assert "checkpoint1" in data["test_context_checkpoint"]
    assert (
        data["test_context_checkpoint"]["checkpoint1"]["elapsed_time"] >= 0.1
    )
    assert (
        data["test_context_checkpoint"]["checkpoint1"]["cumulative_time"]
        >= 0.1
    )


def test_context_manager(temp_files):
    """
    Test case to verify the context manager functionality of TimeWith class.

    Args:
        temp_files (fixture): Fixture providing temporary log and JSON filenames.
    """
    log_filename, json_filename = temp_files

    with TimeWith(
        name="test_context_manager",
        log_filename=log_filename,
        json_filename=json_filename,
    ) as tw:
        time.sleep(0.1)
        tw.checkpoint("checkpoint_cm")

    # Asserts the context manager logs the final elapsed time when exiting the context.
    with open(log_filename, "r") as f:
        logs = f.readlines()
    assert "checkpoint_cm" in logs[-2]
    assert "finished" in logs[-1]

    with open(json_filename, "r") as f:
        data = json.load(f)
    assert "checkpoint_cm" in data["test_context_manager"]
    assert "time" in data["test_context_manager"]
    assert data["test_context_manager"]["checkpoint_cm"]["elapsed_time"] >= 0.1
    assert data["test_context_manager"]["time"] >= 0.1
