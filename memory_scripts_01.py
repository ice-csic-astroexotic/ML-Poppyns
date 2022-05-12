import os
import random

import pandas as pd
import psutil
from memory_profiler import profile

from pypopsyn.simulator.configuration import cfg

process = psutil.Process(os.getpid())
print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)


@profile
def memory_test():
    n = cfg["NS_number"]  # number of records in file
    s = 20  # desired sample size
    filename = "dyn_database_csv_3e6/final_pop_dyn.csv"

    skip = sorted(random.sample(range(2, n + 1), n - s))
    df = pd.read_csv(filename, skiprows=skip, header=[0, 1])

    print("df memory usage in MB:", df.memory_usage().sum() / 1024 / 1024)

    print(df)
    df.to_csv("memory_example_01.csv")

    process = psutil.Process(os.getpid())
    print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)
    return n


if __name__ == "__main__":
    memory_test()
