"""
    Compute dataset statistics.

    This module computes the statistics of the provided dataset.
    In particular it computes the mean, std, max and min values for the labels in the dataset.

    Running the code:

        python3 compute_statistics.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np


def compute_statistics(dataset_dict: dict) -> dict:
    """
    This method computes the label statistics for the provided dataset.
    In particular the mean, std, max and min values for the labels are computed and saved into a dictionary.

    Args:
        dataset_dict (dict): Dictionary containing the information on the dataset.

    Return:
        dict: Dictionary providing the statistical information of each label.

    """

    statistics_dictionary = {}

    # Loop over every key of the training dataset to collect all labels.
    for key, values in dataset_dict.items():
        # If the "input:" prefix is not found in the key name, it is the key of a label.

        if "input:" not in key:

            target_mean = np.mean(values)
            target_std = np.std(values)
            target_max = np.max(values)
            target_min = np.min(values)

            label_statistics = {
                key: {
                    "mean": target_mean,
                    "std": target_std,
                    "max": target_max,
                    "min": target_min,
                }
            }

            # Update the dictionary containing the statistical information.
            statistics_dictionary = {
                **statistics_dictionary,
                **label_statistics,
            }

    return statistics_dictionary
