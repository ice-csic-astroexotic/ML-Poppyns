#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

"""
Random sampling a subset of k lines from a csv file with n lines.


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

random.seed(1)


def choose_rows(
    number_of_rows_to_select,
    total_number_of_rows,
    previously_chosen_rows: Optional[List[int]] = None,
) -> List[int]:

    if previously_chosen_rows is None:
        previously_chosen_rows = []
    sample = random.sample(
        range(1, total_number_of_rows), number_of_rows_to_select
    )
    while (
        len(np.intersect1d(np.array(sample), np.array(previously_chosen_rows)))
        > 0
    ):
        diff = np.setdiff1d(np.array(sample), np.array(previously_chosen_rows))
        new_sample = random.sample(
            range(1, total_number_of_rows),
            number_of_rows_to_select - len(diff),
        )
        concat = np.concatenate([diff, np.array(new_sample)])
        if np.max(np.unique(concat, return_counts=True)[1]) != 1:
            print(" ")
        sample = list(concat)

    return sorted(sample)


def select(
    file_path, k, n, previously_chosen_rows: Optional[List[int]] = None
):
    selected_rows = choose_rows(k, n, previously_chosen_rows)
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
                print("loc", value, selected_rows[i - 1])
                if loc < 0:
                    print(" ")
                data += list(islice(iterator, loc, loc + 1))
        result = [header_1, header_2] + data
    df = pd.read_csv(StringIO("".join(result)), header=[0, 1])
    df = df.set_index(("Unnamed: 0_level_0", "Unnamed: 0_level_1"))
    df.index.name = ""
    return df
