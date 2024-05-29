"""
Test for the memory_efficient_sampling.py module.

    Authors:

        Celsa Pardo (pardo @ ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2022

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

import numpy as np
import pandas as pd

from utilities.memory_efficient_sampling import choose_rows, select


def test_choose_rows():
    """
    Verifying that the returned rows are unique.
    """

    size_subset = 10
    size_full_dataset = 1000
    previously_chosen_rows = random.sample(range(10), 4)

    selected_rows = choose_rows(
        size_subset, size_full_dataset, previously_chosen_rows
    )

    assert np.max(np.unique(selected_rows, return_counts=True)[1]) == 1


def test_select():
    """
    Verifying that this function returns a Dataframe.
    """

    path_file_test = pathlib.Path(
        "data/example_simulation_dyn/final_pop_dyn.csv"
    )

    df = select(path_file_test, 5, 300000)

    assert isinstance(df, pd.DataFrame)
