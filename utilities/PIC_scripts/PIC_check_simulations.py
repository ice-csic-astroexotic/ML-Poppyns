"""
    PIC simulation checker.

    With this script, we count how many of the simulations run using HTCondor have failed.

    We save the directory name of those simulations which failed in a csv called failed_folder.csv.

    Display help message to run the code:

    python PIC_check_simulations.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo (pardo @ csic.es)
"""

import argparse
import os
import pathlib

import pandas as pd


def check_simulations(args):
    """
    Checking how many simulations run using HTCondor have failed.
    We save the name of the output folder for each of the failed simulations in the failed_folders.csv file.

    Args:
        args:
            output_dir_simulation (pathlib.Path): Output directory where the simulation outputs are.

    Returns:
        Nothing.
    """

    output_simulations_path = args.output_dir_simulation
    list_directories = os.listdir(output_simulations_path)

    failed_folder_csv_path = args.dir_failed_folder_csv

    count_error = 0
    fail_simulation = []

    for folder in list_directories:

        simulations_directory = pathlib.Path().joinpath(
            output_simulations_path, folder
        )
        if os.path.isdir(simulations_directory):
            files_in_directory = os.listdir(simulations_directory)

            # If the simulations have finished successfully, then 8 files will be located within each output folder.
            # Hence, by checking the fil number, we can identify those simulations that have failed.
            if len(files_in_directory) < 8:
                count_error += 1
                fail_simulation.append(os.path.basename(simulations_directory))
    df = pd.DataFrame(data={"folder": fail_simulation})
    df.to_csv(
        pathlib.Path().joinpath(failed_folder_csv_path, "failed_folders.csv")
    )


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="HTCondor parameter")

    args.add_argument(
        "--output_dir_simulation",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory with the output from the simulations.",
    )

    args.add_argument(
        "--dir_failed_folder_csv",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory where the csv file containing the list of failed folders is saved.",
    )

    args = args.parse_args()
    check_simulations(args)
