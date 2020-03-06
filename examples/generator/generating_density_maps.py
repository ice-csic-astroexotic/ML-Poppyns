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

import os
import pathlib
import sys

import hydra
import numpy as np
import pandas as pd

import pypopsyn.simulator.configuration as configuration
from pypopsyn.generator.density_map import generate_density_map


@hydra.main()
def generate_density_maps(cfg) -> None:

    n_samples = cfg["n_samples"]
    pop_path = cfg["pop_path"]

    # Since hydra is running we are now in a folder multiran/yyyy-mm-dd/hh-mm-ss/0,
    # therefore we need to move up by three folders to go in multiran
    rebased_path = os.path.normpath(os.getcwd() + 3 * (os.sep + os.pardir))

    # loops over all generated final population samples
    for s in range(n_samples):
        sample_path = (
            rebased_path
            + "/"
            + pop_path
            + "/{0}/final_population.txt".format(s)
        )
        SAMPLE_PATH = pathlib.Path(sample_path)

        # Check if final population file exists as a precondition.
        if not SAMPLE_PATH.exists():
            print("Population file not found in {}".format(SAMPLE_PATH))
            sys.exit()

        df_final = pd.read_csv(SAMPLE_PATH, skiprows=[1])

        generate_density_map(
            df_final["x_final"],
            (-20.0, 20.0),
            df_final["y_final"],
            (-20.0, 20.0),
            "density_map_pop_{0}.png".format(s),
            log_scale=False,
        )


if __name__ == "__main__":
    generate_density_maps()
