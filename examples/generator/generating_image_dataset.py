""" Generator for dataset.

    This module create a dataset of the simulated populations.
    For every population simulated from the simulator `initialize_evolve_population.py`
    with different initial parameters, a density map is created.
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
    """
        This method reads the simulated population files saved in a multirun/date/time
        folder and generates a dataset of density map .png files for each population.
        All the information about the dataset are stored in a datset.csv file
        containing the density map files names and the set of parameter values for
        each simulated population.
    Args:
        args:
            date (str): Date in the form yyyy-mm-dd when the simulated population
            files have been created.

            time (str): Time in the form hh-mm-ss when the simulated population files
            have been created.

            dataset_name (str): Name of the dataset where the density maps will be saved
    """

    # create the dataset directory
    dataset_path = "examples/data/{}/{}/{}".format(
        args.date, args.time, args.dataset_name
    )
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # initialize dictionaries that will contain the density map files names and the
    # corresponding set of parameters values
    density_map_xy_dictionary = {}
    density_map_xz_dictionary = {}
    velocity_map_xy_vr_dictionary = {}
    velocity_map_xy_vphi_dictionary = {}
    velocity_map_xy_vz_dictionary = {}
    param_dictionary = {}

    # Check if the parsed multirun directory exists as a precondition.
    root_path = pathlib.Path("multirun/{}/{}".format(args.date, args.time))
    if not root_path.exists():
        log.error("directory {} not found".format(root_path))
        sys.exit()

    # number of samples in the parsed directory
    sample_number = len(os.listdir(root_path))

    for s in range(sample_number):

        # Check if the simulated population file exists as a precondition.
        pop_path = pathlib.Path(
            "{}/{}/final_population.txt".format(root_path, s)
        )

        if not pop_path.exists():
            log.error("Population file not found in {}".format(pop_path))
            sys.exit()

        # create a data frame object of the population file
        df_pop = pd.read_csv(pop_path, skiprows=[1])

        # create position density maps projected on xy plane
        density_map_xy_filename = "{}/density_map_xy_pop_{}.png".format(
            dataset_path, s
        )

        dg.generate_density_map(
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            density_map_xy_filename,
            n_bins=64,
        )

        # save density map filenames into a dictionary
        density_map_xy_dictionary.setdefault("density_map_xy", []).append(
            density_map_xy_filename
        )

        log.info(
            "xy density map .png generated for sample {} and saved in {}".format(
                s, dataset_path
            )
        )

        # create position density maps projected on xz plane
        density_map_xz_filename = "{}/density_map_xz_pop_{}.png".format(
            dataset_path, s
        )

        dg.generate_density_map(
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["z"],
            (-5.0, 5.0),
            density_map_xz_filename,
            n_bins=64,
        )

        # save density map filenames into a dictionary
        density_map_xz_dictionary.setdefault("density_map_xz", []).append(
            density_map_xz_filename
        )

        log.info(
            "xz density map .png generated for sample {} and saved in {}".format(
                s, dataset_path
            )
        )

        # create velocity maps of component v_r in the xy plane
        velocity_map_xy_vr_filename = "{}/velocity_map_xy_vr_pop_{}.png".format(
            dataset_path, s
        )

        dg.generate_avg_weight_map(
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_r"]),
            velocity_map_xy_vr_filename,
            n_bins=64,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vr_dictionary.setdefault(
            "velocity_map_xy_vr", []
        ).append(velocity_map_xy_vr_filename)

        log.info(
            "vr velocity map in xy plane .png generated for sample {} and saved "
            "in {}".format(s, dataset_path)
        )

        # create velocity maps of component v_phi in the xy plane
        velocity_map_xy_vphi_filename = "{}/velocity_map_xy_vphi_pop_{}.png".format(
            dataset_path, s
        )

        dg.generate_avg_weight_map(
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_phi"]),
            velocity_map_xy_vphi_filename,
            n_bins=64,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vphi_dictionary.setdefault(
            "velocity_map_xy_vphi", []
        ).append(velocity_map_xy_vphi_filename)

        log.info(
            "vphi velocity map in xy plane .png generated for sample {} and saved "
            "in {}".format(s, dataset_path)
        )

        # create velocity maps of component v_z in the xy plane
        velocity_map_xy_vz_filename = "{}/velocity_map_xy_vz_pop_{}.png".format(
            dataset_path, s
        )

        dg.generate_avg_weight_map(
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_z"]),
            velocity_map_xy_vz_filename,
            n_bins=64,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vz_dictionary.setdefault(
            "velocity_map_xy_vz", []
        ).append(velocity_map_xy_vz_filename)

        log.info(
            "vz velocity map in xy plane .png generated for sample {} and saved "
            "in {}".format(s, dataset_path)
        )

        # Check if files containing labels exists as a precondition.
        label_path = pathlib.Path(
            "{}/{}/.hydra/config.yaml".format(root_path, s)
        )

        if not label_path.exists():
            log.error(
                "File containing labels not found in {}".format(label_path)
            )
            sys.exit()

        # save the parameters value in a dictionary
        with open(label_path) as file:
            dictionary = yaml.full_load(file)
            for key, val in dictionary.items():
                param_dictionary.setdefault(key, []).append(val)

    # Merge the filename and parameters dictionaries in a single dictionary
    dataset_dictionary = {
        **density_map_xy_dictionary,
        **density_map_xz_dictionary,
        **velocity_map_xy_vr_dictionary,
        **velocity_map_xy_vphi_dictionary,
        **velocity_map_xy_vz_dictionary,
        **param_dictionary,
    }

    # Write the dataset dictionary into a .csv file
    dataset_filename = "{}/dataset_image.csv".format(dataset_path)
    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)

    log.info("File dataset_image.csv generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--date",
        nargs="?",
        type=str,
        default="2020-03-06",
        help="Date in the form yyyy-mm-dd when the simulated population files have "
        "been created",
    )
    parser.add_argument(
        "--time",
        nargs="?",
        type=str,
        default="12-00-00",
        help="Time in the form hh-mm-ss when the simulated population files have "
        "been created",
    )
    parser.add_argument(
        "--dataset_name",
        nargs="?",
        type=str,
        default="train_set",
        help="Name of the dataset where the density maps will be saved",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
