""" Compute dataset statistics.

    This module compute the statistics of the provided dataset.
    In particular t compute the mean, std, max and min values for the labels in the dataset.

    Running the code:

        python3 compute_statistics.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import numpy as np


def compute_statistics(dataset_dict: dict) -> dict:
    """
    This method compute the label stistics for the provided dataset.
    In particular the mean, std, max and min values for the labels are computed and saved into a disctionary.

    Args:
        dataset_dict (dict): Dictionary containing the information on the dataset.

    Return:
        dict: Dictionary providing the statistical information of each label.

    """

    statistics_dictionary = {}

    # Loop over every key of the train dataset to collect all labels.
    for key, values in dataset_dict.items():
        # If an input prefix is not found, it is a label.

        if "input:" not in key:
            print(key)
            print(values)
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
