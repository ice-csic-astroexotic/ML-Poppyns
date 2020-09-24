import argparse
import json
import typing

fields = {"elapsed_time": "Time [s]", "cumulative_time": "Cumulative [s]"}
accumulate = "elapsed_time"

WIDTH = 20
FLOAT_WIDTH = 8


def generate_header(fields: typing.List) -> str:
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
    multicolumn_str = (
        "| "
        + name
        + " " * (WIDTH * (columns + 1) + columns - len(name) - 2)
        + " |\n"
    )
    return multicolumn_str


def generate_separator(fields: int) -> str:
    separator_str = "+"
    for _ in range(fields + 1):
        separator_str += "-" * WIDTH + "+"
    separator_str += "\n"
    return separator_str


def generate_row(name: str, values: typing.List) -> str:
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
    with open(filename, "r") as f:
        data = json.load(f)

        accumulated_time = 0.0
        table_str = generate_separator(len(fields.items()))
        table_str += generate_header([v for _, v in fields.items()])

        for key, value in data.items():
            table_str += generate_multicolumn(key, len(fields.items()))
            table_str += generate_separator(len(fields.items()))

            for k, v in value.items():
                if k == "time":
                    continue

                accumulated_time += v[accumulate]
                time_values = [x for kk, x in v.items() if kk in fields.keys()]
                table_str += generate_row(k, time_values)
                table_str += generate_separator(len(fields.items()))

        table_str += generate_multicolumn(" ", len(fields.items()))
        table_str += generate_separator(len(fields.items()))
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
