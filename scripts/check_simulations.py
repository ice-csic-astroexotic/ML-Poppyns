"""
    With this script we count how many of the simulations run in HTCondor have failed.

    We save the folder name of those which failed in a csv called failed_folder.csv.


    Authors:
        Celsa Pardo (pardo @ csic.es)


Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import pathlib

import pandas as pd


def check_simulations(args):

    output_simulations_path = args.output_dir_simulation
    list_directories = os.listdir(output_simulations_path)

    count_error = 0
    fail_simulation = []

    for folder in list_directories:

        simulations_directory = pathlib.Path().joinpath(
            output_simulations_path, folder
        )
        files_in_directory = os.listdir(simulations_directory)

        if len(files_in_directory) < 8:
            count_error += 1
            fail_simulation.append(os.path.basename(simulations_directory))

    df = pd.DataFrame(data={"folder": fail_simulation})
    df.to_csv("failed_folders.csv")


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="HTCondor parameter")

    args.add_argument(
        "--output_dir_simulation",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory with the output from the simulations.",
    )

    args = args.parse_args()
    check_simulations(args)
