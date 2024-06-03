"""
Test for the cprofile.py module.

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
