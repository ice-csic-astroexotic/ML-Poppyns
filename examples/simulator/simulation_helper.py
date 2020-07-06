#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Simulator helper script.

    This script helps running the simulator scripts (in this case the simulator
    for initializing and evolving a population) so that we can simplify the way
    parameters are specified for Hydra.

    Our current version of Hydra allows us to perform a parameter sweep but we
    need to provide every single value for each argument to sweep.

    This script parses a more compact representation like:
        --argument low high count

    And expand it to a linspace between [low, high] with a count num of steps.

    It then automatically calls the simulation script.

    Running the code:

        python3 simulator_helper.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import logging
import os
import subprocess
import sys

import numpy as np

log = logging.getLogger(__name__)


def main(args):

    log.info(args)

    cli_args: list = []
    cli_str: list = []

    log.info("Parsing arguments...")

    # Expand each one of the arguments with their linspace. Each argument
    # provides three values in an array: low, high, and number of samples.
    for arg in vars(args):

        log.info(arg)
        log.info(getattr(args, arg))
        arg_range = getattr(args, arg)

        var_range = np.linspace(arg_range[0], arg_range[1], int(arg_range[2]))
        var_str = ",".join(map(str, var_range))
        cli_args.append(arg)
        cli_str.append(var_str)

    log.info("Running simulator...")

    # Generate list for the command which consists of the python interpreter,
    # the script path, and then each one of the arguments with their values,
    # in the end we indicate -m to tell Hydra this is a multirun.
    cmd: list = [
        "python",
        "examples/simulator/initialize_evolve_population.py",
    ]
    for i in range(len(cli_args)):
        cmd.append(cli_args[i] + "=" + cli_str[i])
    cmd.append("-m")

    # Launch simulator with the expaned command.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # Capture all process output and redirect it to the console.
    while True:
        output = process.stdout.readline()
        if output == "" or process.poll() is not None:
            break
        if output:
            print(output.strip().decode("utf-8"))

    log.info("Process finished...")


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn Simulator Helper")

    args.add_argument(
        "--vp_mean",
        nargs=3,
        type=float,
        default=[100.0, 600.0, 6.0],
        help="Range for the mean kick velocity [low, high, steps]",
    )

    args.add_argument(
        "--h_mean",
        nargs=3,
        type=float,
        default=[0.18, 0.18, 1.0],
        help="Range for the mean Z position [low, high, steps]",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
