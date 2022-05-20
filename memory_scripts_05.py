import os
import pathlib
from datetime import datetime
from io import StringIO
from itertools import islice
from math import exp, floor, log
from optparse import Option
from random import random, randrange
from typing import List, Optional

import ipdb
import pandas as pd
import psutil
from memory_profiler import profile

process = psutil.Process(os.getpid())
print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)


def already_removed(selection: str, ignore_index: List[str]):
    """Determine if the selection has already been removed."""
    index = int(selection.split(",")[0])
    if index in ignore_index:
        return True
    return False


def reservoir_sample(
    iterable, k: int = 1, ignore_index: Optional[List[int]] = None
) -> List[str]:
    """Select k items uniformly from iterable.

    Returns the whole population if there are k or fewer items

    from https://bugs.python.org/issue41311#msg373733
    """
    iterator = iter(iterable)
    values = list(islice(iterator, k))
    W = exp(log(random()) / k)
    while True:
        # skip is geometrically distributed
        skip = floor(log(random()) / log(1 - W))
        selection = list(islice(iterator, skip, skip + 1))
        if selection:
            if ignore_index is not None:
                if already_removed(
                    selection=selection[0], ignore_index=ignore_index
                ):
                    continue
            values[randrange(k)] = selection[0]
            W *= exp(log(random()) / k)
        else:
            return values


def sample_file(
    filepath: pathlib.Path, k: int, ignore_index: Optional[List[int]] = None
) -> pd.DataFrame:
    with filepath.open("r") as f:
        header_1 = next(f)
        header_2 = next(f)
        result = [header_1, header_2] + reservoir_sample(f, k, ignore_index)
    df = pd.read_csv(StringIO("".join(result)), header=[0, 1])
    df = df.set_index(("Unnamed: 0_level_0", "Unnamed: 0_level_1"))
    df.index.name = ""
    return df


def filter_df(df: pd.DataFrame):
    selected_df = df[df["array 1"]["[yr]"] < 50.0]
    return selected_df


def main():
    start_time = datetime.now()
    FILE_PATH = pathlib.Path(__file__).parent.joinpath("data_example_B.csv")
    k = 200
    df_1 = sample_file(FILE_PATH, k)
    assert len(df_1) == k
    selected_df_1 = filter_df(df_1)
    df_2 = sample_file(
        FILE_PATH, k, ignore_index=selected_df_1.index.values.tolist()
    )
    selected_df_2 = filter_df(df_2)
    print(selected_df_2)

    end_time = datetime.now()
    print(end_time - start_time)

    return pd.concat([selected_df_1, selected_df_2]).sort_index()


if __name__ == "__main__":
    df = main()
    print(df)
