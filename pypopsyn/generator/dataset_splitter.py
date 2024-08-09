"""
    Splitter for dataset.

    This module splits the provided dataset into training, validation and test datasets
    according to the given split fractions.

    Display help message to run the code:

    python dataset_splitter.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Tuple

import numpy as np
import pandas as pd

import pypopsyn.generator.compute_statistics as cs

log = logging.getLogger(__name__)


def split_dataset(dataset_dict: dict, split: float) -> Tuple[dict, dict]:
    """
    This method splits the provided dataset into two sub-datasets, set_1 and set_2,
    according to a specified split fraction.

    Args:
        dataset_dict (dict): Dictionary containing the information on the dataset.
        split (float): Fraction of set_1 size with respect to the provided dataset size.

    Returns:
        (Tuple[dict, dict]): Two dictionaries providing the information on the two sub-datasets created from the split.
    """

    # Check if the split argument falls in the range (0, 1).
    if (split <= 0.0) or (split >= 1.0):
        log.error(
            f"Split argument {split} out of range. It must be in the range (0, 1)."
        )
        sys.exit()

    dataset_size = [len(x) for x in dataset_dict.values()][0]

    # Evaluate the set_1 size according to the fraction defined by the split argument.
    # Then randomly sample set_1 and set_2 from the whole dataset.
    set1_size = int(split * dataset_size)
    dataset_idx = np.arange(dataset_size)

    set1_idx = np.random.choice(dataset_size, set1_size, replace=False)
    set2_idx = np.array([idx for idx in dataset_idx if idx not in set1_idx])

    # Create dictionaries for both datasets.
    set1_dataset_dictionary = {}
    set2_dataset_dictionary = {}

    for k in dataset_dict.keys():
        v = np.array(dataset_dict[k])
        v_set1 = v[set1_idx]
        v_set2 = v[set2_idx]
        set1_dataset_dictionary.setdefault(k, v_set1)
        set2_dataset_dictionary.setdefault(k, v_set2)

    return set1_dataset_dictionary, set2_dataset_dictionary


def main(args: argparse.Namespace) -> None:
    """
    This function reads a dataset from a specified path, splits it into training,
    validation, and test sets based on the provided split fractions, and saves
    the resulting datasets to CSV files. It also computes statistics for the
    training dataset and saves them to a JSON file.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:
            - dataset_path (str): The path where the simulation dataset is stored.
            - test_split (float or None): The fraction of the total dataset to
              allocate for the test set. Must be in the range [0, 1].
            - valid_split (float or None): The fraction of the training dataset
              to allocate for the validation set. Must be in the range [0, 1].
    """

    dataset_filename = f"{args.dataset_path}/dataset_full.csv"
    dataset_dictionary = pd.read_csv(dataset_filename, header=[0]).to_dict(
        "series"
    )

    if args.test_split is not None and args.valid_split is not None:

        # Split the dataset into training, validation and test sets.
        (
            test_dataset_dictionary,
            trainval_dataset_dictionary,
        ) = split_dataset(dataset_dictionary, args.test_split)

        valid_dataset_dictionary, train_dataset_dictionary = split_dataset(
            trainval_dataset_dictionary, args.valid_split
        )

        # Write the training, validation and test dataset dictionaries into .csv files.
        train_dataset_filename = f"{args.dataset_path}/dataset_train.csv"

        train_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in train_dataset_dictionary.items()
            }
        )
        train_df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

        valid_dataset_filename = f"{args.dataset_path}/dataset_valid.csv"
        valid_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in valid_dataset_dictionary.items()
            }
        )
        valid_df.to_csv(valid_dataset_filename, encoding="utf-8", index=False)

        test_dataset_filename = f"{args.dataset_path}/dataset_test.csv"
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

        # Compute the statistics on the training dataset only.
        statistics_dictionary = cs.compute_statistics(train_dataset_dictionary)

        # Save dictionary containing statistical information to the dataset path in a .json file.
        train_statistics_dump_path = pathlib.Path().joinpath(
            args.dataset_path, "statistics_train.json"
        )
        with open(train_statistics_dump_path, "w") as f:
            json.dump(statistics_dictionary, f, indent=4, sort_keys=True)

        log.info("Files statistics_train.json generated")

    elif args.test_split is None and args.valid_split is not None:

        # Split the dataset into training and validation sets only.
        valid_dataset_dictionary, train_dataset_dictionary = split_dataset(
            dataset_dictionary, args.valid_split
        )

        # Write the training and validation dataset dictionaries into .csv files.
        train_dataset_filename = f"{args.dataset_path}/dataset_train.csv"

        train_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in train_dataset_dictionary.items()
            }
        )
        train_df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

        valid_dataset_filename = f"{args.dataset_path}/dataset_valid.csv"
        valid_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in valid_dataset_dictionary.items()
            }
        )
        valid_df.to_csv(valid_dataset_filename, encoding="utf-8", index=False)

        log.info("Files dataset_train.csv and dataset_valid.csv generated")

        # Compute the statistics on the training dataset only.
        statistics_dictionary = cs.compute_statistics(train_dataset_dictionary)

        # Save dictionary containing statistical information to the dataset path in a .json file.
        train_statistics_dump_path = pathlib.Path().joinpath(
            args.dataset_path, "statistics_train.json"
        )
        with open(train_statistics_dump_path, "w") as f:
            json.dump(statistics_dictionary, f, indent=4, sort_keys=True)

        log.info("Files statistics_train.json generated")

    elif args.test_split is not None and args.valid_split is None:

        # Split the dataset into training and test sets only.
        test_dataset_dictionary, train_dataset_dictionary = split_dataset(
            dataset_dictionary, args.test_split
        )

        # Write the training and validation dataset dictionaries into .csv files.
        train_dataset_filename = f"{args.dataset_path}/dataset_train.csv"

        train_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in train_dataset_dictionary.items()
            }
        )
        train_df.to_csv(train_dataset_filename, encoding="utf-8", index=False)

        test_dataset_filename = f"{args.dataset_path}/dataset_test.csv"
        test_df = pd.DataFrame(
            {
                key: pd.Series(value)
                for key, value in test_dataset_dictionary.items()
            }
        )
        test_df.to_csv(test_dataset_filename, encoding="utf-8", index=False)

        log.info("Files dataset_train.csv and dataset_test.csv generated")

        # Compute the statistics on the training dataset only.
        statistics_dictionary = cs.compute_statistics(train_dataset_dictionary)

        # Save dictionary containing statistical information to the dataset path in a .json file.
        train_statistics_dump_path = pathlib.Path().joinpath(
            args.dataset_path, "statistics_train.json"
        )
        with open(train_statistics_dump_path, "w") as f:
            json.dump(statistics_dictionary, f, indent=4, sort_keys=True)

        log.info("Files statistics_train.json generated")

    else:
        raise ValueError("Please specify a split fraction.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--dataset_path",
        nargs="?",
        type=str,
        default=None,
        help="Path where the simulation dataset is stored.",
    )
    parser.add_argument(
        "--test_split",
        nargs="?",
        type=float,
        default=None,
        help="Fraction of the total simulation dataset that will form the test dataset. "
        "It must be a number in the range [0, 1].",
    )
    parser.add_argument(
        "--valid_split",
        nargs="?",
        type=float,
        default=None,
        help="Fraction of the simulation dataset not dedicated for testing that will form the validation dataset. "
        "It must be a number in the range [0, 1].",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
