import os
import random

import numpy as np
import pandas as pd
import psutil
from memory_profiler import profile

from pypopsyn.simulator.configuration import cfg

process = psutil.Process(os.getpid())
print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)


@profile
def memory_test():
    s = 100  # desired sample size
    filename = "dyn_database_csv_3e6/final_pop_dyn.csv"

    df = pd.read_csv(filename, header=[0, 1], index_col=0)
    df["weights"] = 1.0
    # df.at[1, "weights"] = 0.0
    print(df)

    df_export = pd.DataFrame(columns=df.columns)
    print(df_export)

    print("df memory usage in MB:", df.memory_usage().sum() / 1024 / 1024)

    for i in range(2):
        df_sampled = df.sample(n=s, weights="weights", random_state=1)
        print(df_sampled)
        df_sampled["age"] = df_sampled["age"] * 10
        print(df_sampled)

        df_append = df_sampled[df_sampled["age"]["[yr]"] < 1e8]
        index = df_append.index.values
        print(index)

        df_export = pd.concat([df_export, df_append])
        print(df_export)

        df.at[index, "weights"] = 0.0
        print(df)

    df_export.to_csv("memory_example_04.csv")

    process = psutil.Process(os.getpid())
    print("Memory usage in MB:", process.memory_info().rss / 1024 / 1024)


if __name__ == "__main__":
    memory_test()
