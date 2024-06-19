"""
    Test for the timefunc module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import os
import tempfile
import time
from unittest.mock import patch

import pytest

from utilities.benchmark.timefunc import time_function


def test_time_function_decorator():
    """
    Test the time_function decorator when printing to terminal is disabled.
    """
    # Simulate a function that takes some time to execute.
    @time_function(show=False)
    def dummy_function(x):
        time.sleep(0.1)
        return x * 2

    result = dummy_function(2)

    # Assert that the function returns the expected result.
    assert result == 4


@patch("builtins.print")
def test_time_function_show(mock_print):
    """
    Test that the time_function decorator prints the timing information to the console
    when the show parameter is set to True.

    Args:
        mock_print (MagicMock): A MagicMock object representing the print function.
        It allows assertions on how the print function is called within the decorated code.
    """

    @time_function(show=True)
    def dummy_function(x):
        time.sleep(0.1)
        return x * 2

    dummy_function(2)

    # Assert that the timing information is printed on terminal.
    assert mock_print.called
    # Assert that the printed message contains the function name and timing information.
    assert "dummy_function took" in mock_print.call_args[0][0]


def test_time_function_file():
    """
    Test that the time_function decorator writes the timing information to a specified file
    when the filename parameter is provided.
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name

    try:

        @time_function(filename=filename, show=False)
        def dummy_function(x):
            time.sleep(0.1)
            return x * 2

        # Call the decorated function to ensure it writes timing information to the file.
        dummy_function(2)

        # Read the content of the temporary file after the function has been called.
        with open(filename, "r") as f:
            content = f.read()

        assert "dummy_function took" in content

    finally:
        # Clean up the temporary file.
        os.remove(filename)
