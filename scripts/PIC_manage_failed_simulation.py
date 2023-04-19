"""
    Managing of failed simulations at PIC.

    After the failed simulations have been launched again and finished successfully,
    we use this script to transfer the new output back to the original folders.

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
import shutil


def manage_failed_simulations(args):
    """
    Copying the successfully relaunched output back to the original folders.

    Args:
        args:
            simulation_dir (pathlib.Path): Output directory where the simulation outputs are.
            failed_simulation_dir (pathlib.Path): Output directory where the failed simulation outputs are.

    Returns:
        Nothing.
    """

    output_simulations_path = args.simulation_dir
    failed_simulation_path = args.failed_simulation_dir

    output_failed_simulations = pathlib.Path().joinpath(
        failed_simulation_path, "output_simulations"
    )

    list_directories_ = os.listdir(output_failed_simulations)

    for folder in list_directories_:

        folder_failed_path = pathlib.Path().joinpath(
            output_failed_simulations, folder
        )
        folder_original_path = pathlib.Path().joinpath(
            output_simulations_path, folder
        )

        # We avoid to move the override.json to prevent permissions issues.
        # Copying these files is not needed as they are the same as the original ones.
        for files in os.listdir(folder_failed_path):
            if files != "override.json":
                files_path = pathlib.Path().joinpath(folder_failed_path, files)
                files_original_path = pathlib.Path().joinpath(
                    folder_original_path, files
                )
                shutil.copy(files_path, files_original_path)


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="HTCondor parameter")

    args.add_argument(
        "--failed_simulation_dir",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory containing the output from rerunning the failed simulations.",
    )

    args.add_argument(
        "--simulation_dir",
        nargs="?",
        type=str,
        default=None,
        help="Path to the directory with the original output from the simulations.",
    )

    args = args.parse_args()
    manage_failed_simulations(args)
