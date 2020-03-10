""" Generator example for density map.

    This is an example of how to use the density map generation function from
    the generator subpackage to create a heatmap of data simulated from the
    simulator subpackage using the `generating_population.py` sample script.
    This expects that an initial population has been generated in
    `examples/data/initia_population.txt`.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import os
import pathlib
import sys

import pandas as pd

from pypopsyn.generator.density_map import generate_density_map


def generate_density_maps(args) -> None:
    pathlib.Path("examples/generator/" + args.dataset_name).mkdir(
        parents=True, exist_ok=True
    )

    sample_index = 0

    exclude = set([".hydra"])
    for root, dirs, files in os.walk(args.root_path):
        dirs[:] = [d for d in dirs if d not in exclude]
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            file_path = pathlib.Path(os.path.join(dir_path, args.file_name))
            print(file_path)

            # Check if final population file exists as a precondition.
            if not file_path.exists():
                print("Population file not found in {}".format(file_path))
                sys.exit()

            df_final = pd.read_csv(file_path, skiprows=[1])

            generate_density_map(
                df_final["x_final"],
                (-20.0, 20.0),
                df_final["y_final"],
                (-20.0, 20.0),
                "examples/generator/"
                + args.dataset_name
                + "/density_map_pop_{0}.png".format(sample_index),
                log_scale=False,
            )

            sample_index += 1


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
    generate_density_maps(args)
