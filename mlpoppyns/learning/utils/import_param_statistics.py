"""
    Module containing tools to load the statistics of labels of a dataset in a list.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import json

import numpy as np


def import_statistics(stats_path: str):
    """
    Extracting the mean, standard deviation, minimum and maximum values for all the parameters in the `stats_path` file.

    Args:
        stats_path (str): Path to the file where the statistics are saved.

    Returns:
        (torch.tensor, torch.tensor): Mean and standard deviation for the parameters in the `stats_path` file.
    """

    mean_list = []
    std_list = []
    max_list = []
    min_list = []

    with open(stats_path, "r") as json_file:
        data = json.load(json_file)

    for key, value in data.items():
        mean_list.append(value["mean"])
        std_list.append(value["std"])
        max_list.append(value["max"])
        min_list.append(value["min"])

    mean_list = np.array(mean_list)
    std_list = np.array(std_list)
    max_list = np.array(max_list)
    min_list = np.array(min_list)

    return mean_list, std_list, max_list, min_list
