""" Generator for dataset.

    This module creates a dataset of the simulated populations for every
    population simulated from the simulator `initialize_evolve_population.py`
    with different initial parameters.

    This expects that a population has been generated using hydra multirun method.

    The user can choose to generate either a dataset of images or of 2D arrays.

    The information on the simulated dataset is also saved in a .csv file where the
    corresponding input files are mapped with their paths and labels.

    Running the code:

        python3 dataset_generator.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia@ice.csic.es)

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
        folder and generates a dataset of density map array .npy files for each
        population.
        All the information about the dataset are stored in a datset.csv file
        containing the density map files names and the set of parameter values for
        each simulated population.
    Args:
        args:
            date (str): Date in the form yyyy-mm-dd when the simulated population
            files have been created.

            time (str): Time in the form hh-mm-ss when the simulated population files
            have been created.

            dataset_name (str): Name of the dataset where the density maps matrices
            will be saved.

            type (str): Type of dataset to generate: array or image.

            resolution (int): Resolution (number of bins per axis for the 2d
            histograms) for the image to generate.
    """

    # create the dataset directory
    dataset_path = "examples/data/{}".format(args.dataset_name)
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # Initialize the multiple options we have to generate the different data
    # inputs which will be later selected at runtime depending on the arguments.
    position_map_generators = {
        "array": dg.generate_density_matrix,
        "image": dg.generate_density_map,
    }
    velocity_map_generators = {
        "array": dg.generate_avg_weight_matrix,
        "image": dg.generate_avg_weight_map,
    }
    extensions = {"array": "npy", "image": "png"}

    # initialize dictionaries that will contain the density map files names and the
    # corresponding set of parameters values
    position_map_xy_dictionary = {}
    position_map_xz_dictionary = {}
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

        log.info("Generating sample {}".format(s))

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
        position_map_xy_filename = "{}/position_map_xy_pop_{}.{}".format(
            dataset_path, s, extensions[args.type]
        )

        position_map_generators[args.type](
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            position_map_xy_filename,
            n_bins=args.resolution,
            normalize=args.normalize,
        )

        # save density map filenames into a dictionary
        position_map_xy_dictionary.setdefault(
            "input:position_map_xy", []
        ).append(position_map_xy_filename)

        log.info("{} generated...".format(position_map_xy_filename))

        # create position density maps projected on xz plane
        position_map_xz_filename = "{}/position_map_xz_pop_{}.{}".format(
            dataset_path, s, extensions[args.type]
        )

        position_map_generators[args.type](
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["z"],
            (-5.0, 5.0),
            position_map_xz_filename,
            n_bins=args.resolution,
            normalize=args.normalize,
        )

        # save density map filenames into a dictionary
        position_map_xz_dictionary.setdefault(
            "input:position_map_xz", []
        ).append(position_map_xz_filename)

        log.info("{} generated...".format(position_map_xz_filename))

        # create velocity maps of component v_r in the xy plane
        velocity_map_xy_vr_filename = "{}/velocity_map_xy_vr_pop_{}.{}".format(
            dataset_path, s, extensions[args.type]
        )

        velocity_map_generators[args.type](
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_r"]),
            velocity_map_xy_vr_filename,
            n_bins=args.resolution,
            normalize=args.normalize,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vr_dictionary.setdefault(
            "input:velocity_map_xy_vr", []
        ).append(velocity_map_xy_vr_filename)

        log.info("{} generated...".format(velocity_map_xy_vr_filename))

        # create velocity maps of component v_phi in the xy plane
        velocity_map_xy_vphi_filename = "{}/velocity_map_xy_vphi_pop_{}.{}".format(
            dataset_path, s, extensions[args.type]
        )

        velocity_map_generators[args.type](
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_phi"]),
            velocity_map_xy_vphi_filename,
            n_bins=args.resolution,
            normalize=args.normalize,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vphi_dictionary.setdefault(
            "input:velocity_map_xy_vphi", []
        ).append(velocity_map_xy_vphi_filename)

        log.info("{} generated...".format(velocity_map_xy_vphi_filename))

        # create velocity maps of component v_z in the xy plane
        velocity_map_xy_vz_filename = "{}/velocity_map_xy_vz_pop_{}.{}".format(
            dataset_path, s, extensions[args.type]
        )

        velocity_map_generators[args.type](
            df_pop["x"],
            (-20.0, 20.0),
            df_pop["y"],
            (-20.0, 20.0),
            abs(df_pop["v_z"]),
            velocity_map_xy_vz_filename,
            n_bins=args.resolution,
            normalize=args.normalize,
        )

        # save velocity map filenames into a dictionary
        velocity_map_xy_vz_dictionary.setdefault(
            "input:velocity_map_xy_vz", []
        ).append(velocity_map_xy_vz_filename)

        log.info("{} generated...".format(velocity_map_xy_vz_filename))

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
        **position_map_xy_dictionary,
        **position_map_xz_dictionary,
        **velocity_map_xy_vr_dictionary,
        **velocity_map_xy_vphi_dictionary,
        **velocity_map_xy_vz_dictionary,
        **param_dictionary,
    }

    # Write the dataset dictionary into a .csv file
    dataset_filename = "{}/dataset.csv".format(dataset_path)
    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)

    log.info("File dataset.csv generated")


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
        default="array_train_set",
        help="Name of the dataset where the density maps will be saved",
    )
    parser.add_argument(
        "--type",
        nargs="?",
        type=str,
        default="array",
        help="Type of dataset to generate: array or image",
    )
    parser.add_argument(
        "--resolution",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the arrays that will be generated (in number of cells)",
    )
    parser.add_argument(
        "--normalize",
        nargs="?",
        type=bool,
        default=True,
        help="Generate normalized maps or not",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
