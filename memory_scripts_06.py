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
        sample = list(np.concatenate([diff, np.array(new_sample)]))

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
                data += list(islice(iterator, loc, loc + 1))
        result = [header_1, header_2] + data
    df = pd.read_csv(StringIO("".join(result)), header=[0, 1])
    df = df.set_index(("Unnamed: 0_level_0", "Unnamed: 0_level_1"))
    df.index.name = ""
    return df


def filter_df(df: pd.DataFrame):
    """Test filter function to select on the sampled values"""
    selected_df = df[df["array 1"]["[yr]"] < 50.0]
    return selected_df


def main():
    """Run through 2 sample select steps and apply the filter"""
    FILE_PATH = pathlib.Path(__file__).parent.joinpath("data_example.csv")
    k = 100  # how many items to sample
    row_count = 10000  # total number of data rows in the data file

    df_1 = select(FILE_PATH, k, row_count)
    assert df_1.index.is_unique

    selected_df_1 = filter_df(df_1)
    print("selected 1:", selected_df_1)

    df_2 = select(FILE_PATH, k, row_count, selected_df_1.index.values.tolist())
    assert df_2.index.is_unique

    selected_df_2 = filter_df(df_2)
    print("selected 2:", selected_df_1)

    combined_df = pd.concat([selected_df_1, selected_df_2]).sort_index()

    print(combined_df[combined_df.index.value_counts() > 1])
    assert combined_df.index.is_unique

    print("selected combined:", combined_df)


if __name__ == "__main__":
    main()
