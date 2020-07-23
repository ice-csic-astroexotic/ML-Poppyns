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
import json
import logging
import subprocess
import sys

import numpy as np

log = logging.getLogger(__name__)


def main(args):
    log.info(args)

    log.info("Parsing arguments...")

    cli_args: list = []
    cli_str: list = []

    args_dict = vars(args)

    # Open the parameters dictionary with required values for the selection ones.
    f = open("examples/simulator/config_sweeper.json")
    check_arg = json.load(f)

    required_parameters = []
    forbidden_parameters = []

    for arg in vars(args):

        log.info(arg)
        value = getattr(args, arg)
        log.info(value)

        if value is None:
            continue

        elif type(value) is str:
            # If the value of this parameter is a string, this is a selection
            # parameter and we must check: (a) whether the selection is valid
            # (b) capture the list of required parameters and (c) gather the
            # forbidden ones (probably they belong other selection).
            if value in check_arg[arg]:
                cli_args.append(arg)
                cli_str.append(value)
                required_parameters.extend(check_arg[arg][value])
                forbidden_parameters.extend(
                    [
                        item
                        for sublist in [
                            v for k, v in check_arg[arg].items() if k != value
                        ]
                        for item in sublist
                    ]
                )
            else:
                # If the value for such argument is not on the dictionary of
                # possible values we throw an exception.
                raise ValueError(
                    "The value {} is not feasible for parameter {}".format(
                        value, arg
                    )
                )

        elif type(value) is list:
            # If the value is a list, we assume it will be a specificaiton of
            # three values [low, high, steps] and then expand each one of the
            # arguments with the linear space in such range.
            var_range = np.linspace(value[0], value[1], int(value[2]))
            var_str = ",".join(map(str, var_range))
            cli_args.append(arg)
            cli_str.append(var_str)

    log.info("Required parameters {}".format(required_parameters))
    log.info("Forbidden parameters {}".format(forbidden_parameters))

    # Remove intersecting parameters from the forbidden list.
    for p in set(required_parameters) & set(forbidden_parameters):
        log.info("Intersecting parameter {}".format(p))
        forbidden_parameters.remove(p)
    # Check if all the required parameters are specified.
    for p in required_parameters:
        if p not in args_dict.keys() or args_dict[p] is None:
            raise ValueError("Required parameter {} not present.".format(p))
    # Check if none of the incompatible parameters are required.
    for p in forbidden_parameters:
        if p in args_dict.keys() and args_dict[p] is not None:
            raise ValueError("Forbidden parameter {} is present".format(p))

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
