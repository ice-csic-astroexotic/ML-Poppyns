""" Generator example for density map.

    This is an example of how to use the density map generation function from
    the generator subpackage to create a heatmap of data simulated from the
    simulator subpackage using the `generating_population.py` sample script.
    This expects that an initial population has been generated in
    `examples/data/initia_population.txt`.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import pandas as pd

from pypopsyn.generator.density_map import generate_density_map

if __name__ == "__main__":

    df_initial = pd.read_csv(
        "./examples/data/initial_population.txt", skiprows=[1]
    )
    generate_density_map(
        df_initial["x_initial"],
        (np.min(df_initial["x_initial"]), np.max(df_initial["x_initial"])),
        df_initial["y_initial"],
        (np.min(df_initial["y_initial"]), np.max(df_initial["y_initial"])),
        "example_density_map.png",
        log_scale=False,
    )
