"""
    Tests for the configuration_parser module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

from argparse import ArgumentTypeError

import pytest

import pypopsyn.learning.configuration_parser as cp


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        # Test valid string inputs
        ("True", True),
        ("true", True),
        ("1", True),
        ("False", False),
        ("false", False),
        ("0", False),
        # Test None (mimicking --flag without a value)
        (None, True),
    ],
)
def test_str_to_bool_valid(input_value, expected_output):
    """
    Test valid cases for str_to_bool.
    """
    assert cp.str_to_bool(input_value) == expected_output


def test_str_to_bool_invalid():
    """
    Test invalid input that should raise ArgumentTypeError.
    """
    with pytest.raises(ArgumentTypeError):
        cp.str_to_bool("invalid")
