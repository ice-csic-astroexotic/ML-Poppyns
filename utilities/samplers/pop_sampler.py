"""
    Population sampler script.

    This script randomly samples a population with a reduced number of stars for every
    population simulated from the simulator `initialize_evolve_population.py`
    with different initial parameters.

    This expects that a set of populations have been generated either using
    directly that script or using the helper with its particular directory tree.

    The user can choose the total number of stars to randomly select from the
    original simulated population, they can provide a weighted selection according to
    the stars' distances from the Sun and can provide a distance cut-off to select only stars
    that are nearer to the Sun.

    The resampled population is also saved in a .pkl.gz file alongside with the .json
    file containing the related labels.

    Display help message to run the code:

    python pop_sampler.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def calculate_selection_weights(d: np.ndarray) -> np.ndarray:
    """
    Calculate the weights to assign to every star for selection.
    Weights are evaluated as a function of distance from the Sun, nearest stars are easier
    and more likely to be detected.

    Args:
        d (np.ndarray): Array of distances from the Sun [kpc].

    Returns:
        (np.ndarray): Array of selection weights.
    """

    # This function has been fine-tuned to match the distribution of distances from the Sun of
    # the 224 neutron stars with observed proper motion. In this sample we selected neutron stars
    # that are likely to be not recycled and isolated (i.e., with a spin period derivative Pdot>10^(-17)
    # and with no association to globular clusters or binary systems).
    weights = np.exp(-0.5 * d) / d

    # Normalize the weights to their sum.
    w = weights / np.sum(weights)

    return w


def data_sampler(args: argparse.Namespace) -> None:
    """
    This function reads the simulated population files (usually by the simulation
    helper) folder and creates simulated population files with a reduced number
    of stars by randomly sampling the original evolved population file.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - data (str): Path to where the simulated populations are located.
            - save_dir (str): Path to where to save the resampled population files.
            - size (int): Number of stars to randomly sample from the population files.
            - distance_cut (float): Maximum distance from the Sun cut-off.
            - uniform (bool): If True stars are selected uniformly in distance from the simulated population.
    """

    # Check if the parsed simulated populations' directory exists.
    root_path = pathlib.Path(args.data)
    if not root_path.exists():
        log.error(f"Directory {root_path} not found...")
        sys.exit()

    # Number of samples in the parsed directory.
    sample_number = len(os.listdir(root_path))

    for s in range(sample_number):

        # Create the resampled data directory path.
        data_path = f"{args.save_dir}/{s:06}"
        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

        log.info(f"Resampling sample {s:06}")

        # Check if the simulated population file exists as a precondition.
        pop_path = pathlib.Path(f"{root_path}/{s:06}/final_population.pkl.gz")

        if not pop_path.exists():
            log.error(f"Population file not found in {pop_path}")
            sys.exit()

        # Check if files containing labels exists as a precondition.
        label_path = pathlib.Path(f"{root_path}/{s:06}/override.json")

        if not label_path.exists():
            log.error(f"File containing labels not found in {label_path}")
            sys.exit()

        with open(label_path) as f:
            override = json.load(f)

        # Save the file containing labels into the new directory path.
        override_dump_path = pathlib.Path().joinpath(
            f"{data_path}/", "override.json"
        )
        with open(override_dump_path, "w") as f:
            json.dump(override, f, indent=4, sort_keys=True)

        # Create a data frame object of the population file.
        df_pop = pd.read_pickle(pop_path, compression="gzip")

        # If a distance cut-off is provided, select only stars in the solar neighborhood.
        if args.distance_cut is not None:
            df_pop = df_pop[df_pop["d"]["[kpc]"] < args.distance_cut]

            if args.uniform:
                # Select stars randomly from the simulated population.
                df_select = df_pop.sample(int(np.floor(args.size)))

            else:
                # Select stars according to some weights that are function of the distance from the Sun.
                w = calculate_selection_weights(
                    df_pop["d"]["[kpc]"].to_numpy()
                )
                df_select = df_pop.sample(args.size, replace=False, weights=w)

        else:

            log.error("Please define a distance cut.")
            break

        # Save the resampled data frame as compressed binary file.
        output_path = f"{data_path}/final_population.pkl.gz"
        df_select.to_pickle(output_path, compression="gzip")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--data",
        nargs="?",
        type=str,
        required=True,
        help="Path to where the simulated populations are.",
    )
    parser.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        required=True,
        help="Path to where to save the resampled population files.",
    )
    parser.add_argument(
        "--size",
        nargs="?",
        type=int,
        default=None,
        help="Number of stars to randomly sample from the population files.",
    )
    parser.add_argument(
        "--distance_cut",
        nargs="?",
        type=float,
        default=None,
        help="Maximum distance from the Sun cut-off.",
    )
    parser.add_argument(
        "--uniform",
        dest="uniform",
        default=False,
        action="store_true",
        help="If True stars are selected uniformly in distance from the simulated population.",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_sampler(args)
