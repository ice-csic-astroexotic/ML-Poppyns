import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
from memory_profiler import profile

from pypopsyn.simulator.configuration import cfg

process = psutil.Process(os.getpid())
print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)


@profile
def parquet_test():
    start_time = datetime.now()
    array = np.linspace(0, 100, 21)

    column_index_1 = [
        "array 1",
        "array 2",
    ]
    column_index_2 = [
        "[yr]",
        "[kpc]",
    ]
    header = pd.MultiIndex.from_arrays([column_index_1, column_index_2])

    df = pd.DataFrame(
        data=np.array([array, array]).T,
        columns=header,
    )

    print(df)

    df.to_parquet("parquet_example_01.parquet", engine="pyarrow")
    df.to_csv("parquet_example_01.csv")

    process = psutil.Process(os.getpid())
    print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)

    end_time = datetime.now()
    print("Duration: {}".format(end_time - start_time))


if __name__ == "__main__":
    parquet_test()
