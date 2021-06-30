""" Generator for dataset.

    This module creates a dataset of the simulated populations for every
    population simulated from the simulator `initialize_evolve_population.py`
    with different initial parameters.

    This expects that a set of populations have been generated either using
    directly that script or using the helper with its particular directory tree.

    The user can choose to generate either a dataset of images or of 2D arrays.

    The information on the simulated dataset is also saved in a .csv file where the
    corresponding input files are mapped with their paths and labels.

    If the argument split is specified the total dataset will be split into a
    train and validation subsets and two additional .csv files will be created
    specifying the samples in each subset.

    Running the code:

        python3 dataset_generator.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd

import pypopsyn.generator.compute_statistics as cs
import pypopsyn.generator.dataset_splitter as ds
import pypopsyn.generator.position_maps as pmaps
import pypopsyn.generator.ppdot_maps as ppdmaps
import pypopsyn.generator.velocity_maps as vmaps
import pypopsyn.simulator.basics.constants as const

log = logging.getLogger(__name__)


def generate_dataset(args) -> None:
    """
    This method reads the simulated population files (usually by the simulation
    helper) folder and generates a dataset of density maps in the specified
    format (images or arrays) and with a specified resolution.
    All the information about the dataset are stored in a dataset.csv file
    containing the density map files names and the set of parameter values for
    each simulated population.

    Args:
        args:
            data (str): Path to where the simulated populations are located.

            save_dir (str): Path to where the generated dataset will be saved.

            type (str): Type of dataset to generate: array or image.

            resolution_dyn (int): Resolution (number of bins per axis for the 2d
            histograms) for the position and velocity maps to generate. In case of RA DEC maps the
            DEC axis has half the number of bins with respect to the RA axis.

            resolution_ppdot (int): Resolution (number of bins per axis for the 2d
            histograms) for the P-Pdot density maps to generate.
    """

    # Create the dataset directory path.
    dataset_path = f"{args.save_dir}"
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
    ppdot_map_dictionary = {}
    param_dictionary = {}

    # Check if the parsed simulated populations directory exists.
    root_path = pathlib.Path(args.data)
    if not root_path.exists():
        log.error(f"Directory {root_path} not found...")
        sys.exit()

    # If the dataset split is provided check if the argument falls in the range [0, 1].
    if args.test_split and (
        (args.test_split <= 0.0) or (args.test_split >= 1.0)
    ):
        log.error(
            f"Split argument {args.test_split} out of range. It must be in the range (0, 1)."
        )
        sys.exit()

    if args.valid_train_split and (
        (args.valid_train_split <= 0.0) or (args.valid_train_split >= 1.0)
    ):
        log.error(
            f"Split argument {args.valid_train_split} out of range. It must be in the range (0, 1)."
        )
        sys.exit()

    # Number of samples in the parsed directory.
    sample_number = len(os.listdir(root_path))

    log.info(f"Generating {sample_number} samples...")

    # Main generator loop.
    for s in range(sample_number):

        log.info(f"Generating sample {s:06}")

        # Check if the simulated population file exists as a precondition.
        pop_path = pathlib.Path(f"{root_path}/{s:06}/final_population.pkl.gz")

        if not pop_path.exists():
            log.error(f"Population file not found in {pop_path}")
            sys.exit()

        # Create a data frame object of the population file.
        df_pop = pd.read_pickle(str(pop_path), compression="gzip")

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
            args.resolution_dyn,
            args.resolution_dyn,
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
            args.resolution_dyn,
            args.resolution_dyn,
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
            args.resolution_dyn,
            args.resolution_dyn,
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
            args.resolution_dyn,
            args.resolution_dyn,
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
            args.resolution_dyn,
            args.resolution_dyn,
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
            args.resolution_dyn,
            int(args.resolution_dyn / 2),
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
            args.resolution_dyn,
            int(args.resolution_dyn / 2),
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
            args.resolution_dyn,
            int(args.resolution_dyn / 2),
            velocity_map_vdec_dictionary,
            x_limits=(0.0, 360.0),
            y_limits=(-90.0, 90.0),
        )

        # Create P-Pdot density maps.
        ppdmaps.generate_ppdot_map(
            dataset_path,
            "ppdot_map",
            s,
            args.type,
            df_pop["P"],
            df_pop["P_dot"] / const.YR_TO_S,
            args.resolution_ppdot,
            args.resolution_ppdot,
            ppdot_map_dictionary,
        )

        # Check if files containing labels exists as a precondition.
        label_path = pathlib.Path(f"{root_path}/{s:06}/override.json")

        if not label_path.exists():
            log.error(f"File containing labels not found in {label_path}")
            sys.exit()

        # Save the parameters value in a dictionary.
        with open(label_path) as file:
            dictionary = json.load(file)
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
        **ppdot_map_dictionary,
        **param_dictionary,
    }

    # Write the whole dataset dictionary into a .csv file.
    dataset_filename = f"{dataset_path}/dataset.csv"

    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)

    log.info("File dataset.csv generated")

    if args.test_split is not None and args.valid_train_split is not None:

        (
            test_dataset_dictionary,
            trainval_dataset_dictionary,
        ) = ds.split_dataset(dataset_dictionary, args.test_split)
        valid_dataset_dictionary, train_dataset_dictionary = ds.split_dataset(
            trainval_dataset_dictionary, args.valid_train_split
        )

        # Write the training, validation and test dataset dictionaries into .csv files.
        train_dataset_filename = f"{dataset_path}/dataset_train.csv"

        train_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in train_dataset_dictionary.items()
            }
        )
        train_df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

        valid_dataset_filename = f"{dataset_path}/dataset_valid.csv"
        valid_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in valid_dataset_dictionary.items()
            }
        )
        valid_df.to_csv(valid_dataset_filename, encoding="utf-8", index=False)

        test_dataset_filename = f"{dataset_path}/dataset_test.csv"
        test_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in test_dataset_dictionary.items()
            }
        )
        test_df.to_csv(test_dataset_filename, encoding="utf-8", index=False)

        log.info(
            "Files dataset_train.csv, dataset_valid.csv and dataset_test.csv generated"
        )

        # Compute the statistics on the train dataset only.

        statistics_dictionary = cs.compute_statistics(train_dataset_dictionary)
        # Save dictionary containing statistical information to the dataset path in a .json file.
        train_statistics_dump_path = pathlib.Path().joinpath(
            dataset_path, "statistics_train.json"
        )
        with open(train_statistics_dump_path, "w") as f:
            json.dump(statistics_dictionary, f, indent=4, sort_keys=True)

        log.info("Files statistics_train.json generated")

    if args.test_split is None and args.valid_train_split is not None:

        valid_dataset_dictionary, train_dataset_dictionary = ds.split_dataset(
            dataset_dictionary, args.valid_train_split
        )

        # Write the training and validation dataset dictionaries into .csv files.
        train_dataset_filename = f"{dataset_path}/dataset_train.csv"

        train_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in train_dataset_dictionary.items()
            }
        )
        train_df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

        valid_dataset_filename = f"{dataset_path}/dataset_valid.csv"
        valid_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in valid_dataset_dictionary.items()
            }
        )
        valid_df.to_csv(valid_dataset_filename, encoding="utf-8", index=False)

        log.info("Files dataset_train.csv and dataset_valid.csv generated")

        # Compute the statistics on the train dataset only.

        statistics_dictionary = cs.compute_statistics(train_dataset_dictionary)
        # Save dictionary containing statistical information to the dataset path in a .json file.
        train_statistics_dump_path = pathlib.Path().joinpath(
            dataset_path, "statistics_train.json"
        )
        with open(train_statistics_dump_path, "w") as f:
            json.dump(statistics_dictionary, f, indent=4, sort_keys=True)

        log.info("Files statistics_train.json generated")


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
        "--test_split",
        nargs="?",
        type=float,
        default=None,
        help="Fraction of the total dataset that will form the test dataset. "
        "It must be a number in the range [0, 1].",
    )
    parser.add_argument(
        "--valid_train_split",
        nargs="?",
        type=float,
        default=None,
        help="Fraction of the train dataset that will form the validation dataset. "
        "It must be a number in the range [0, 1].",
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
        "--resolution_dyn",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the position and velocity maps that will be generated (in number of cells).",
    )
    parser.add_argument(
        "--resolution_ppdot",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the P-Pdot maps that will be generated (in number of cells).",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
