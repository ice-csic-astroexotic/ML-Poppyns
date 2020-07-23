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
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2020
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

import argparse
import logging
import os
import subprocess
import sys

import numpy as np

log = logging.getLogger(__name__)


def main(args):
    log.info(args)

    log.info("Parsing arguments...")

    cli_args: list = []
    cli_str: list = []

    # If the arg values are a in a list providing three values in an array [low, high, and number of samples],
    # expand each one of the arguments with their linspace. Otherwise just save the value of the arg.
    for arg in vars(args):
        log.info(arg)
        log.info(getattr(args, arg))
        values = getattr(args, arg)

        if values is None:
            continue
        elif type(values) is not list:
            cli_args.append(arg)
            cli_str.append(values)
        elif type(values) is list:
            var_range = np.linspace(values[0], values[1], int(values[2]))
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

    # Launch simulator with the expanded command.
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
    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--kick_model",
        nargs="?",
        type=str,
        default=None,
        help="pdf model for the kick velocity and range for its parameter. Choose between km_exp or km_maxwell.",
    )

    args.add_argument(
        "--sigma_k",
        nargs=3,
        type=float,
        default=None,
        help="Range of kick velocity sigma for the Maxwell model [low, high, steps]",
    )

    args.add_argument(
        "--vk_c",
        nargs=3,
        type=float,
        default=None,
        help="Range of characteristic kick velocity for the exponential model [low, high, steps]",
    )

    args.add_argument(
        "--h_c",
        nargs=3,
        type=float,
        default=[0.18, 0.18, 1.0],
        help="Range for the mean Z position [low, high, steps]",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
