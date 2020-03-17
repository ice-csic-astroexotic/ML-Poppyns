""" Generator for dataset.

    This module create a dataset of the simulated populations.
    For every population simulated from the simulator `generate_evolve_population.py` with different initial parameters, a density map is created.
    This expects that a population has been generated using hydra multirun method.
    The simulated dataset is also saved in a .csv file where the density map .png
    files and the correponding parameters are saved.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import logging
import os
import pathlib
import sys

import pandas as pd
import yaml

import pypopsyn.generator.dataset_generator as dg

log = logging.getLogger(__name__)


def generate_dataset(args) -> None:
    pathlib.Path("examples/data/" + args.dataset_name).mkdir(
        parents=True, exist_ok=True
    )

    sample_index = 0
    filename_dictionary = {}
    param_dictionary = {}

    for root, dirs, files in os.walk(args.root_path):
        # os.walk() returns the list of directories in arbitrary order, therefore
        # sorting them is needed to give the correct association between .png
        # density map filename and label
        dirs.sort()

        for file_name in files:
            # search for the population file
            if file_name == args.file_name:
                pop_path = pathlib.Path(os.path.join(root, args.file_name))

                # Check if the population files exists as a precondition.
                if not pop_path.exists():
                    log.error(
                        "Population file not found in {}".format(pop_path)
                    )
                    sys.exit()

                # generate density map
                log.info(
                    "Generating density map .png for sample {0} and saving in "
                    "examples/data/{1}".format(sample_index, args.dataset_name)
                )

                df_pop = pd.read_csv(pop_path, skiprows=[1])

                density_map_filename = "examples/data/{0}/density_map_pop_{1}.png".format(
                    args.dataset_name, sample_index
                )

                dg.generate_density_map(
                    df_pop["x_final"],
                    (-20.0, 20.0),
                    df_pop["y_final"],
                    (-20.0, 20.0),
                    density_map_filename,
                    log_scale=False,
                )

                # save filenames into a dictionary
                filename_dictionary.setdefault("filename", []).append(
                    density_map_filename
                )

                sample_index += 1

            # search for the label file
            elif file_name == "config.yaml":
                label_path = pathlib.Path(os.path.join(root, "config.yaml"))

                # Check if files containing labels exists as a precondition.
                if not label_path.exists():
                    log.error(
                        "File containing labels not found in {}".format(
                            label_path
                        )
                    )
                    sys.exit()

                # save the parameters in a dictionary
                with open(label_path) as file:
                    dictionary = yaml.full_load(file)
                    for key, val in dictionary.items():
                        param_dictionary.setdefault(key, []).append(val)

    # Merge the filename and parameters dictionaries in a single dictionary
    dataset_dictionary = {**filename_dictionary, **param_dictionary}

    # Write the dataset dictionary into a .csv file
    dataset_filename = "examples/data/{0}/dataset.csv".format(
        args.dataset_name
    )
    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--root_path",
        nargs="?",
        type=str,
        default="multirun/2020-03-06/17-16-12",
        help="Path to the directory where the simulated populations "
        "are stored",
    )
    parser.add_argument(
        "--file_name",
        nargs="?",
        type=str,
        default="final_population.txt",
        help="Simulated population file name",
    )
    parser.add_argument(
        "--dataset_name",
        nargs="?",
        type=str,
        default="train_set",
        help="Name of the dataset where the density maps are saved",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
