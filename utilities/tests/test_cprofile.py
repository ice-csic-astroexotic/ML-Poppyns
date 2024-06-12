"""
    Test for the cprofile.py module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import tempfile
from pathlib import Path

import pytest

from utilities.benchmark.cprofile import do_cprofile


# Mock function to be profiled.
def mock_function():
    for _ in range(1000):
        pass


def test_do_cprofile_enabled():
    """
    Test the cprofile method when profiling is enabled.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Decorate the mock function with profiling enabled.
        profiled_function = do_cprofile(enabled=True, output_dir=temp_dir)(
            mock_function
        )

        # Call the decorated function.
        profiled_function()

        # Check that the profiling result file is created.
        profile_file = Path(temp_dir) / "mock_function.txt"
        assert profile_file.is_file(), "Profile file was not created"

        # Check that the file is not empty.
        assert profile_file.stat().st_size > 0, "Profile file is empty"


def test_do_cprofile_disabled():
    """
    Test the cprofile method when profiling is disabled.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Decorate the mock function with profiling disabled.
        profiled_function = do_cprofile(enabled=False, output_dir=temp_dir)(
            mock_function
        )

        # Call the decorated function.
        profiled_function()

        # Check that no profiling result file is created.
        profile_file = Path(temp_dir) / "mock_function.txt"
        assert not profile_file.is_file(), "Profile file should not be created"
