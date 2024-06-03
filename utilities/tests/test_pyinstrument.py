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

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from utilities.benchmark.pyinstrument import profile


def test_profile_enabled_show_output(capsys):
    """
    Test that the profile decorator works correctly when profiling is enabled
    and output is shown on terminal.

    Args:
        capsys (pytest fixture): Captures the output to stdout and stderr.
    """

    @profile(enabled=True, show=True)
    def sample_function():
        return "test"

    with patch("pyinstrument.Profiler") as MockProfiler:
        # Mock the Profiler instance methods.
        mock_profiler = MockProfiler.return_value
        mock_profiler.output_text.return_value = "profile output"

        result = sample_function()

        # Assert that the function returns the expected result.
        assert result == "test"

        # Capture and assert that the profile output was printed to stdout.
        captured = capsys.readouterr()
        assert "profile output" in captured.out


def test_profile_enabled_with_output_dir():
    """
    Test that the profile decorator works correctly when profiling is enabled
    and output is written to a specified directory.
    """

    with patch("pyinstrument.Profiler") as MockProfiler:
        # Mock the Profiler instance methods.
        mock_profiler = MockProfiler.return_value
        mock_profiler.output_text.return_value = "profile output"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the output directory path.
            output_path = pathlib.Path(temp_dir) / "output_dir"

            @profile(enabled=True, show=False, output_dir=str(output_path))
            def sample_function():
                return "test"

            sample_function()

            # Construct the expected output file path.
            output_file = output_path / "sample_function.pyinstrument"

            # Assert that the output file was created.
            assert output_file.exists()

            # Assert that the content of the output file is correct.
            with open(output_file, "r") as f:
                content = f.read()
                assert "profile output" in content


def test_profile_disabled():
    """
    Test that the profile decorator works correctly when profiling is disabled.
    """

    @profile(enabled=False)
    def sample_function():
        return "test"

    with patch("pyinstrument.Profiler") as MockProfiler:
        result = sample_function()

        # Assert that the Profiler was not called
        MockProfiler.assert_not_called()

        # Assert that the function returns the expected result.
        assert result == "test"
