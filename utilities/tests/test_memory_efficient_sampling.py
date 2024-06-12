"""
    Test for the memory_efficient_sampling.py module.

    Authors:

        Celsa Pardo (pardo @ ice.csic.es)
"""
import pathlib
import random

import numpy as np
import pandas as pd

from utilities.samplers.memory_efficient_sampling import choose_rows, select


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
