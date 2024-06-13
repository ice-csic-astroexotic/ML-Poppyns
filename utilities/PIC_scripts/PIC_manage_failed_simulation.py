"""
    Managing of failed simulations at PIC.

    After the failed simulations have been launched again and finished successfully,
    we use this script to transfer the new output back to the original folders.

    Display help message to run the code:

    python PIC_manage_failed_simulation.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo (pardo @ csic.es)
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
