#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" JSON Profile to RST Table.

    This script parses a timing profile from TimeWith contexts and generates an
    RST table representation to include in our documentation automagically.

    Running the code:

        python3 scripts/json_profile_to_rst_table.py --filename profile.json

        or

        python3 scripts/json_profile_to_rst_table.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2020
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

import argparse
import json
import typing

fields = {"elapsed_time": "Time [s]", "cumulative_time": "Cumulative [s]"}
accumulate = "elapsed_time"

WIDTH = 20
FLOAT_WIDTH = 8


def generate_header(fields: typing.List) -> str:
    """
    Generates a header for the table providing a list of field names with
    automatic width handling.

    Args:
        fields (List): Names for the fields (columns) excluding the first one.

    Returns:
        An RST string representation for the table header.
    """
    name = "Context"
    row_str = "| " + name + " " * (WIDTH - len(name) - 2) + " |"
    for field in fields:
        row_str += " " + field + " " * (WIDTH - len(field) - 2) + " |"
    row_str += "\n"
    row_str += "+"
    for _ in range(len(fields) + 1):
        row_str += "=" * WIDTH + "+"
    row_str += "\n"
    return row_str


def generate_multicolumn(name: str, columns: int) -> str:
    """
    Generates a multicolumn spawning a number of specified fields and populated
    with a given name and automatic width handling.

    Args:
        name (str): Text to populate the row.
        columns (int): Number of fields or columns to take.

    Returns:
        An RST string representation of the multicolumn.
    """
    multicolumn_str = (
        "| "
        + name
        + " " * (WIDTH * (columns + 1) + columns - len(name) - 2)
        + " |\n"
    )
    return multicolumn_str


def generate_separator(fields: int) -> str:
    """
    Generates a separator for the table with automatic width for it.

    Args:
        fields: Number of columns or fields (excluding the checkpoint names).

    Returns:
        An RST string containing the separator representation.
    """
    separator_str = "+"
    for _ in range(fields + 1):
        separator_str += "-" * WIDTH + "+"
    separator_str += "\n"
    return separator_str


def generate_row(name: str, values: typing.List) -> str:
    """
    Generates a row for the table with the given checkpoint name and the values
    for the fields (which spawn one column each one). The width is automatically
    adjusted to a maximum WIDTH per column.

    Args:
        name (str): Name of the checkpoint (row).
        values (List): Values for each field in the checkpoint (columns).

    Returns:
        An RST string representation of the row with newline at the end.
    """
    row_str = "| " + name + " " * (WIDTH - len(name) - 2) + " |"
    for v in values:
        row_str += (
            " "
            + ("{:" + str(FLOAT_WIDTH) + ".4f}").format(v)
            + " " * (WIDTH - FLOAT_WIDTH - 2)
            + " |"
        )
    row_str += "\n"
    return row_str


def print_table(filename: str) -> None:
    """
    Prints a table in RST format by parsing the specified JSON profile file.

    Args:
        filename (str): Path to the JSON profile to parse.

    Returns:
        Nothing, prints the RST table to the screen.
    """

    with open(filename, "r") as f:
        data = json.load(f)

        # Accumulate total time field.
        accumulated_time = 0.0

        # Add a separator and the header with the selected field names.
        table_str = generate_separator(len(fields.items()))
        table_str += generate_header([v for _, v in fields.items()])

        for key, value in data.items():
            # For each context, place a multicolumn with its name.
            table_str += generate_multicolumn(key, len(fields.items()))
            table_str += generate_separator(len(fields.items()))

            # Traverse each checkpoint of the context.
            for k, v in value.items():

                if k == "time":
                    continue

                # Accumulate this checkpoint total time.
                accumulated_time += v[accumulate]

                # Get the values for the tracked fields in the table and insert
                # a row with the name of the checkpoint and the values for those
                # selected fields.
                time_values = [x for kk, x in v.items() if kk in fields.keys()]
                table_str += generate_row(k, time_values)
                table_str += generate_separator(len(fields.items()))

        # Insert an empty multicolumn to separate total time.
        table_str += generate_multicolumn(" ", len(fields.items()))
        table_str += generate_separator(len(fields.items()))

        # Insert total time row.
        table_str += generate_multicolumn(
            (
                "Total "
                + fields[accumulate]
                + ": "
                + ("{:" + str(FLOAT_WIDTH) + ".4f}").format(accumulated_time)
            ),
            len(fields.items()),
        )
        table_str += generate_separator(len(fields.items()))

        print(table_str)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--filename",
        nargs="?",
        type=str,
        default=None,
        help="Path to the JSON profile to tabulate",
    )

    args = args.parse_args()

    print_table(args.filename)
