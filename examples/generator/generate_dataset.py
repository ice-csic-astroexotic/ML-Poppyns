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

import numpy as np
import pandas as pd
import yaml

import pypopsyn.generator.position_maps as pmaps
import pypopsyn.generator.velocity_maps as vmaps

log = logging.getLogger(__name__)


def generate_dataset(args) -> None:
    """
    This method reads the simulated population files saved in a multirun
    folder and generates a dataset of density maps in the specified format
    (images or arrays) and with a specified resolution.
    All the information about the dataset are stored in a datset.csv file
    containing the density map files names and the set of parameter values for
    each simulated population.

    Args:
        args:
            data (str): Path to where the simulated data in a multirun are located.

            save_dir (str): Path to where the generated dataset will be saved.

            type (str): Type of dataset to generate: array or image.

            resolution (int): Resolution (number of bins per axis for the 2d
            histograms) for the image to generate. In case of RA DEC maps the
            DEC axis has half the number of bins with respect to the RA axis.

            samples (int): Number of samples to generate. If no samples are
                specified the whole dataset is generated. Samples are taken
                equally spaced.
    """

    # Create the dataset directory path.
    dataset_path = "{}".format(args.save_dir)
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # Initialize dictionaries that will contain the density map files names and
    # the corresponding set of parameters values.
    position_map_xy_dictionary = {}
    position_map_xz_dictionary = {}
    position_map_radec_dictionary = {}
    velocity_map_xy_vr_dictionary = {}
    velocity_map_xy_vphi_dictionary = {}
    velocity_map_xy_vz_dictionary = {}
    velocity_map_vra_dictionary = {}
    velocity_map_vdec_dictionary = {}
    param_dictionary = {}

    # Check if the parsed multirun directory exists as a precondition.
    root_path = pathlib.Path(args.data)
    if not root_path.exists():
        log.error("directory {} not found".format(root_path))
        sys.exit()

    # Number of samples in the parsed directory.
    sample_number = len(os.listdir(root_path))

    # Select samples to run.
    samples = []
    if args.samples:
        # If a number of samples is specified, uniformly sample them.
        samples = list(
            np.round(np.linspace(0, sample_number - 1, args.samples)).astype(
                int
            )
        )
    else:
        # If no samples are specified, just generate all of them.
        samples = [i for i in range(sample_number)]

    log.info("Generating {} samples".format(len(samples)))

    # Main generator loop.
    for s in samples:

        log.info("Generating sample {}".format(s))

        # Check if the simulated population file exists as a precondition.
        pop_path = pathlib.Path(
            "{}/{}/final_population.pkl.gz".format(root_path, s)
        )

        if not pop_path.exists():
            log.error("Population file not found in {}".format(pop_path))
            sys.exit()

        # Create a data frame object of the population file.
        df_pop = pd.read_pickle(pop_path, compression="gzip")

        # Remove the units header row from the data frame.
        df_pop.columns = [x[0] for x in df_pop.columns]

        # Create position density maps projected on XY plane.
        pmaps.generate_position_map(
            dataset_path,
            "position_map_xy",
            s,
            args.type,
            df_pop["x"],
            df_pop["y"],
            args.resolution,
            args.resolution,
            position_map_xy_dictionary,
        )

        # Create position density maps projected on XZ plane.
        pmaps.generate_position_map(
            dataset_path,
            "position_map_xz",
            s,
            args.type,
            df_pop["x"],
            df_pop["z"],
            args.resolution,
            args.resolution,
            position_map_xz_dictionary,
        )

        # Create velocity maps of component v_r in the XY plane.
        vmaps.generate_velocity_map(
            dataset_path,
            "velocity_map_xy_vr",
            s,
            args.type,
            df_pop["x"],
            df_pop["y"],
            abs(df_pop["v_r"]),
            args.resolution,
            args.resolution,
            velocity_map_xy_vr_dictionary,
        )

        # Create velocity maps of component v_phi in the XY plane.
        vmaps.generate_velocity_map(
            dataset_path,
            "velocity_map_xy_vphi",
            s,
            args.type,
            df_pop["x"],
            df_pop["y"],
            abs(df_pop["v_phi"]),
            args.resolution,
            args.resolution,
            velocity_map_xy_vphi_dictionary,
        )

        # Create velocity maps of component v_z in the XY plane.
        vmaps.generate_velocity_map(
            dataset_path,
            "velocity_map_xy_vz",
            s,
            args.type,
            df_pop["x"],
            df_pop["y"],
            abs(df_pop["v_z"]),
            args.resolution,
            args.resolution,
            velocity_map_xy_vz_dictionary,
        )

        # Create position density maps projected on RA DEC plane.
        pmaps.generate_position_map(
            dataset_path,
            "position_map_radec",
            s,
            args.type,
            df_pop["RA"],
            df_pop["DEC"],
            args.resolution,
            int(args.resolution / 2),
            position_map_radec_dictionary,
            x_limits=(0.0, 360.0),
            y_limits=(-90.0, 90.0),
        )

        # Create velocity maps of component v_RA in the RA DEC plane.
        vmaps.generate_velocity_map(
            dataset_path,
            "velocity_map_vra",
            s,
            args.type,
            df_pop["RA"],
            df_pop["DEC"],
            abs(df_pop["v_RA"]),
            args.resolution,
            int(args.resolution / 2),
            velocity_map_vra_dictionary,
            x_limits=(0.0, 360.0),
            y_limits=(-90.0, 90.0),
        )

        # Create velocity maps of component v_DEC in the RA DEC plane.
        vmaps.generate_velocity_map(
            dataset_path,
            "velocity_map_vdec",
            s,
            args.type,
            df_pop["RA"],
            df_pop["DEC"],
            abs(df_pop["v_DEC"]),
            args.resolution,
            int(args.resolution / 2),
            velocity_map_vdec_dictionary,
            x_limits=(0.0, 360.0),
            y_limits=(-90.0, 90.0),
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

        # Save the parameters value in a dictionary.
        with open(label_path) as file:
            dictionary = yaml.full_load(file)
            for key, val in dictionary.items():
                param_dictionary.setdefault(key, []).append(val)

    # Merge the filename and parameters dictionaries in a single dictionary.
    dataset_dictionary = {
        **position_map_xy_dictionary,
        **position_map_xz_dictionary,
        **position_map_radec_dictionary,
        **velocity_map_xy_vr_dictionary,
        **velocity_map_xy_vphi_dictionary,
        **velocity_map_xy_vz_dictionary,
        **velocity_map_vra_dictionary,
        **velocity_map_vdec_dictionary,
        **param_dictionary,
    }

    # Evaluate the validation dataset size according to the fraction defined by the split argument.
    # Then random sample the validation dataset and the train dataset from the whole dataset
    valid_size = int(args.split * len(samples))
    dataset_idx = np.arange(len(samples))
    valid_idx = np.random.choice(len(samples), valid_size, replace=False)
    train_idx = np.array([idx for idx in dataset_idx if idx not in valid_idx])
    # Create dictionaries for the training and validation datasets
    valid_dataset_dictionary = {}
    train_dataset_dictionary = {}
    for k in dataset_dictionary.keys():
        v = np.array(dataset_dictionary[k])
        v_valid = v[valid_idx]
        v_train = v[train_idx]
        valid_dataset_dictionary.setdefault(k, v_valid)
        train_dataset_dictionary.setdefault(k, v_train)

    # Write the train and validation dataset dictionary into a .csv file.
    train_dataset_filename = "{}/train_dataset.csv".format(dataset_path)
    df = pd.DataFrame(
        {
            key: pd.Series(value)
            for key, value in train_dataset_dictionary.items()
        }
    )
    df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

    valid_dataset_filename = "{}/valid_dataset.csv".format(dataset_path)
    df = pd.DataFrame(
        {
            key: pd.Series(value)
            for key, value in valid_dataset_dictionary.items()
        }
    )
    df.to_csv(valid_dataset_filename, encoding="utf-8", index=False)

    log.info("Files train_dataset.csv and valid_dataset.csv generated")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--data",
        nargs="?",
        type=str,
        required=True,
        help="Path to where the simulated data in a multirun are.",
    )
    parser.add_argument(
        "--split",
        nargs="?",
        type=float,
        default=0.2,
        help="Fraction of the total dataset that will form the validation dataset.",
    )
    parser.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="examples/data/array_train_set",
        help="Path to the folder where the dataset will be saved.",
    )
    parser.add_argument(
        "--type",
        nargs="?",
        type=str,
        choices=["array", "image"],
        default="array",
        help="Type of dataset to generate: array or image.",
    )
    parser.add_argument(
        "--resolution",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the arrays that will be generated (in number of cells).",
    )
    parser.add_argument(
        "--samples", nargs="?", type=int, help="Number of samples to select",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
