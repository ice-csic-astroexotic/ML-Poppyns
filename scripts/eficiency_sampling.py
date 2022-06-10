#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

"""
Sampling a random subset from a csv file.


    Authors:

        Vanessa Graber (graber @ ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2020
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

import pathlib
import random
from io import StringIO
from itertools import islice
from multiprocessing.context import assert_spawning
from typing import List, Optional

import numpy as np
import pandas as pd
from memory_profiler import profile

random.seed(1)


def choose_rows(
    number_of_rows_to_select: int,
    total_number_of_rows: int,
    previously_chosen_rows: Optional[List[int]] = None,
) -> List[int]:

    """
    Choose a subset of random indexes from all the indexes of a dataframe.

    Args:

        number_of_rows_to_select (pathlib.Path): Number of rows to randomly selected from the full dataset.
        total_number_of_rows (int) : Number of rows in the full dataset.
        previously_chosen_rows (pathlib.Path): Rows previously chosen for the previous subset.

    Returns:

        A sorted list of the randomly chosen indexes.
    """

    if previously_chosen_rows is None:
        previously_chosen_rows = []

    # We remove from the list of indexes those that were previously chosen.
    data_set = np.setdiff1d(
        np.arange(total_number_of_rows), np.array(previously_chosen_rows)
    )

    sample = random.sample(data_set.tolist(), number_of_rows_to_select)

    return sorted(sample)


def select(
    file_path: pathlib.Path,
    size_subset: int,
    size_full_dataset: int,
    previously_chosen_rows: Optional[List[int]] = None,
):

    """
    Select a random subset from a dataset.

    Args:

        file_path (pathlib.Path): Path to the full dataset.
        size_full_dataset (int): Number of rows of the desired random subset.
        size_subset (int) : Number of rows in the full dataset.
        previously_chosen_rows (list): Rows previously chosen for the previous subset.

    Returns:

        Dataframe of the random subset.
    """

    selected_rows = choose_rows(
        size_subset, size_full_dataset, previously_chosen_rows
    )

    data = []

    with file_path.open("r") as f:
        header_1 = f.readline()
        header_2 = f.readline()
        iterator = iter(f)

        for i, value in enumerate(selected_rows):

            if i == 0:
                data += list(islice(iterator, value, value + 1))
            else:
                loc = value - selected_rows[i - 1] - 1

                data += list(islice(iterator, loc, loc + 1))

        result = [header_1, header_2] + data

    df = pd.read_csv(StringIO("".join(result)), header=[0, 1])
    df = df.set_index(("Unnamed: 0_level_0", "Unnamed: 0_level_1"))
    df.index.name = ""
    return df
