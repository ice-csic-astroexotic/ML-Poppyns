"""
Test for the json_profile_to_rst_table.py module.

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

import json
from io import StringIO
from unittest.mock import patch

import pytest

from scripts.json_profile_to_rst_table import (
    generate_header,
    generate_multicolumn,
    generate_row,
    generate_separator,
    print_table,
)


@pytest.fixture
def test_case_1():
    # Sample JSON data for testing
    data = {
        "context1": {
            "checkpoint1": {"elapsed_time": 0.1, "cumulative_time": 0.1},
            "checkpoint2": {"elapsed_time": 0.2, "cumulative_time": 0.3},
        },
        "context2": {
            "checkpoint3": {"elapsed_time": 0.3, "cumulative_time": 0.3},
            "checkpoint4": {"elapsed_time": 0.4, "cumulative_time": 0.7},
        },
    }

    return data


def test_generate_header():
    """
    Test generate_header function.
    """
    expected_result = (
        "| Context            | Time [s]           | Cumulative [s]     |\n"
        "+====================+====================+====================+\n"
    )
    assert generate_header(["Time [s]", "Cumulative [s]"]) == expected_result


def test_generate_multicolumn():
    """
    Test generate_multicolumn function.
    """
    expected_result = "| Context name                                                                      |\n"
    assert generate_multicolumn("Context name", 3) == expected_result


def test_generate_separator():
    """
    Test generate_separator function.
    """
    expected_result = (
        "+--------------------+--------------------+--------------------+\n"
    )
    assert generate_separator(2) == expected_result


def test_generate_row():
    """
    Test generate_row function.
    """
    expected_result = (
        "| Context            |   0.1000           |   0.1000           |\n"
    )
    assert generate_row("Context", [0.1, 0.1]) == expected_result


@patch("builtins.open")
def test_print_table(mock_open, test_case_1):
    """
    Test print_table function.
    """
    # The @patch("builtins.open") decorator is used to replaces the open function used in the print_table
    # function with a MagicMock object.
    mock_open.return_value = StringIO(json.dumps(test_case_1))
    expected_output = (
        "+--------------------+--------------------+--------------------+\n"
        "| Context            | Time [s]           | Cumulative [s]     |\n"
        "+====================+====================+====================+\n"
        "| context1                                                     |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| checkpoint1        |   0.1000           |   0.1000           |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| checkpoint2        |   0.2000           |   0.3000           |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| context2                                                     |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| checkpoint3        |   0.3000           |   0.3000           |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| checkpoint4        |   0.4000           |   0.7000           |\n"
        "+--------------------+--------------------+--------------------+\n"
        "|                                                              |\n"
        "+--------------------+--------------------+--------------------+\n"
        "| Total Time [s]:   1.0000                                     |\n"
        "+--------------------+--------------------+--------------------+\n"
        "\n"
    )
    # We patch the sys.stdout object, redirecting standard output to a StringIO object.
    # This allows capturing the output that would normally be printed to terminal during
    # the execution of the print_table function.
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        print_table("dummy_filename.json")
        assert mock_stdout.getvalue() == expected_output
