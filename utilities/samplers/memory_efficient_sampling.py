"""
    Sampling a random subset from a csv file without loading the full dataset into memory.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import pathlib
import random
import time
from io import StringIO
from itertools import islice
from typing import List, Optional

import numpy as np
import pandas as pd

from pypopsyn.simulator.config_simulator import cfg

# Initialize seed randomly if no seed was specified.
if cfg["seed_sampling"] is None:
    cfg["seed_sampling"] = int(time.time())

random.seed(cfg["seed_sampling"])


def choose_rows(
    number_of_rows_to_select: int,
    total_number_of_rows: int,
    previously_chosen_rows: Optional[List[int]] = None,
) -> List[int]:

    """
    Choose a subset of random indices from all the indices of a dataframe.

    Args:

        number_of_rows_to_select (int): Number of rows to randomly select from the full dataset without taking into
        account the headers.
        total_number_of_rows (int) : Number of rows in the full dataset without taking into account the headers.
        previously_chosen_rows (list): Rows previously chosen from previous subset.

    Returns:

        (list): A sorted list of the randomly chosen indices.
    """

    if previously_chosen_rows is None:
        previously_chosen_rows = []

    # We remove from the list of indices those that were previously chosen as we want to obtain a unique sample
    # of indices.
    data_set = np.setdiff1d(
        np.arange(total_number_of_rows), np.array(previously_chosen_rows)
    )

    # Select the desired number of indices and sample randomly.
    sample = random.sample(data_set.tolist(), number_of_rows_to_select)

    sample_sorted = sorted(sample)

    return sample_sorted


def select(
    file_path: pathlib.Path,
    size_subset: int,
    size_full_dataset: int,
    previously_chosen_rows: Optional[List[int]] = None,
) -> pd.DataFrame:

    """
    Select a random subset from a dataset without loading the full file into memory.
    The following implementation only works when the full dataset has two headers as our `final_pop_dyn.csv`, it won't
    work otherwise.
    Args:

        file_path (pathlib.Path): Path to the full dataset.
        size_subset (int): Number of rows of the desired random subset without taking into account the headers.
        size_full_dataset (int): Number of rows in the full dataset without taking into account the headers.
        previously_chosen_rows (list): Rows previously chosen from previous subset.

    Returns:

        (Dataframe): Dataframe of the random subset.

    """

    selected_rows = choose_rows(
        size_subset, size_full_dataset, previously_chosen_rows
    )

    # Creating an empty list where the chosen rows will be appended.
    data = []

    # Reading the file using an iterator tool to allow for step by step iteration through the data.
    with file_path.open("r") as f:

        # We separately read our two header lines to only iterate through data.
        header_1 = f.readline()
        header_2 = f.readline()
        iterator = iter(f)

        for i, value in enumerate(selected_rows):

            # Iterating through our iterable data file to reach the rows selected previously.
            # We use the islice function to extract the respective data rows individually.
            if i == 0:
                data += list(islice(iterator, value, value + 1))
            else:
                loc = value - selected_rows[i - 1] - 1

                data += list(islice(iterator, loc, loc + 1))

        result = [header_1, header_2] + data

    # Saving the sampled data rows as a pandas DataFrame.
    df = pd.read_csv(StringIO("".join(result)), header=[0, 1])

    # Redefining the data header to allow compatibility with the original format.
    df = df.set_index(("Unnamed: 0_level_0", "Unnamed: 0_level_1"))
    df.index.name = ""

    return df
