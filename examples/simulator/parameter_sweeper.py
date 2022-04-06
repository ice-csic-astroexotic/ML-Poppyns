#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

"""
    parameter sweeper script.

    This script generates the files necessary to launch multiple simulations with different parameter values
    on HTCondor.
    If the --sampling_type argument is set to "grid" it parses a compact representation
    for each tunable parameter like:

        --argument low high count

    And expands it to a linspace between [low, high] with a count num of steps.
    If the --sampling_type argument is set to "random" it parses a compact representation
    for each tunable parameter like:

        --argument low high

    And expands it to a list of values between [low, high] drawn from a uniform distribution.
    In this case the number of values to be drawn for each parameter is specified by the
    argument --sampling_size.
    Such expansion is done for each specified argument and then a generator produces all the possible
    combinations of them if in "grid" mode or sets of random drawn parameter values if in "random" mode.
    Each set will be saved in a JSON parameter override file that will be used as input to a simulation.

    Running the code:

        python3 parameter_sweeper.py --h
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
import itertools
import json
import logging
import multiprocessing as mp
import pathlib
import subprocess
import sys
import threading
import typing

import numpy as np

from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


def set_default(args_dict: dict) -> None:  # noqa: C901
    """
    If any of the parameters related to the simulation are None, set them to the
    default value provided in the configuration file.
    Args:
            args_dict: dictionary of the parsed argument via CLI.
    """

    if args_dict["sigma_k"] is None:
        if args_dict["kick_model"] == "km_maxwell":
            if args_dict["sampling_type"] == "grid":
                args_dict["sigma_k"] = [cfg["sigma_k"], cfg["sigma_k"], 1]
            elif args_dict["sampling_type"] == "random":
                args_dict["sigma_k"] = [cfg["sigma_k"], cfg["sigma_k"]]

            log.info(
                "sigma_k set to the default value {}".format(cfg["sigma_k"])
            )

    if args_dict["vk_c"] is None:
        if args_dict["kick_model"] == "km_exp":
            if args_dict["sampling_type"] == "grid":
                args_dict["vk_c"] = [cfg["vk_c"], cfg["vk_c"], 1]
            elif args_dict["sampling_type"] == "random":
                args_dict["vk_c"] = [cfg["vk_c"], cfg["vk_c"]]

            log.info("vk_c set to the default value {}".format(cfg["vk_c"]))

    if args_dict["h_c"] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict["h_c"] = [cfg["h_c"], cfg["h_c"], 1]
        elif args_dict["sampling_type"] == "random":
            args_dict["h_c"] = [cfg["h_c"], cfg["h_c"]]

        log.info("vk_c set to the default value {}".format(cfg["h_c"]))

    if args_dict["P_initial_mean"] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict["P_initial_mean"] = [
                cfg["P_initial_mean"],
                cfg["P_initial_mean"],
                1,
            ]
        elif args_dict["sampling_type"] == "random":
            args_dict["P_initial_mean"] = [
                cfg["P_initial_mean"],
                cfg["P_initial_mean"],
            ]

        log.info(
            "P_initial_mean set to the default value {}".format(
                cfg["P_initial_mean"]
            )
        )

    if args_dict["P_initial_sigma"] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict["P_initial_sigma"] = [
                cfg["P_initial_sigma"],
                cfg["P_initial_sigma"],
                1,
            ]
        elif args_dict["sampling_type"] == "random":
            args_dict["P_initial_sigma"] = [
                cfg["P_initial_sigma"],
                cfg["P_initial_sigma"],
            ]

        log.info(
            "P_initial_sigma set to the default value {}".format(
                cfg["P_initial_sigma"]
            )
        )

    if args_dict["B_initial_log10_mean"] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict["B_initial_log10_mean"] = [
                cfg["B_initial_log10_mean"],
                cfg["B_initial_log10_mean"],
                1,
            ]
        elif args_dict["sampling_type"] == "random":
            args_dict["B_initial_log10_mean"] = [
                cfg["B_initial_log10_mean"],
                cfg["B_initial_log10_mean"],
            ]

        log.info(
            "B_initial_log10_mean set to the default value {}".format(
                cfg["B_initial_log10_mean"]
            )
        )

    if args_dict["B_initial_log10_sigma"] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict["B_initial_log10_sigma"] = [
                cfg["B_initial_log10_sigma"],
                cfg["B_initial_log10_sigma"],
                1,
            ]
        elif args_dict["sampling_type"] == "random":
            args_dict["B_initial_log10_sigma"] = [
                cfg["B_initial_log10_sigma"],
                cfg["B_initial_log10_sigma"],
            ]

        log.info(
            "B_initial_log10_sigma set to the default value {}".format(
                cfg["B_initial_log10_sigma"]
            )
        )


def check_expand_args(args_dict: dict) -> (list, list):
    """
        Check if the parsed input arguments are coherent and have the correct shape.
        If in grid mode: expand each simulation parameter in linear space in the specified ranges.
        If in random mode: draw random set of parameter values from uniform distributions in the specified ranges.
        Args:
            args_dict: dictionary of the parsed argument via CLI.
        Return:
            (list, list): a list containing the expanded ranges of the parameters and a list containing the names of the
            expanded parameters.
        """
    cli_args: list = []
    cli_str: list = []

    # Open the parameters dictionary with required values for the selection ones.
    f = open("examples/simulator/config_sweeper.json")
    check_arg = json.load(f)

    required_parameters = []
    forbidden_parameters = []

    var_names = []
    var_expanded_ranges = []

    for arg in args_dict.keys():

        log.info(arg)
        value = args_dict[arg]
        log.info(value)

        if value is None:
            if arg == "sampling_size":
                if args_dict["sampling_type"] == "random":
                    raise ValueError(
                        "In random mode you have to specify the parameter sampling_size."
                    )
            else:
                continue

        elif type(value) is str:
            # If the value of this parameter is a string, this can be either
            # the directory path where to save the multirun output or a selection parameter.
            # In this last case we must check: (a) whether the selection is valid
            # (b) capture the list of required parameters and (c) gather
            # the forbidden ones (probably they belong other selection).
            if arg == "output_dir":
                cli_args.append("--output_dir")
                cli_str.append(value)
            elif value in check_arg[arg]:
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
                    f"The value {value} is not feasible for parameter {arg}"
                )

        elif type(value) is list:
            # If the value is a list, we assume it will be a specification of
            # three values if sampling_type = grid or two values if sampling_type = random.

            if args_dict["sampling_type"] == "grid":
                # The three values [low, high, steps] are used to expand each one of the
                # argument with the linear space in the range [low, high] with a number of specified steps.
                if len(value) != 3:
                    raise ValueError(
                        f"In grid mode the list must have length 3 for parameter {arg}"
                    )
                var_range = np.linspace(value[0], value[1], int(value[2]))
                var_expanded_ranges.append(list(var_range))
                var_names.append(arg)

            if args_dict["sampling_type"] == "random":
                # The two values [low, high] define the range from which a number of values
                # (specified according to the sampling_size argument) is drawn from a uniform distribution.
                if len(value) != 2:
                    raise ValueError(
                        f"In random mode the list must have length 2 for parameter {arg}"
                    )
                var_range = np.random.uniform(
                    value[0], value[1], int(args_dict["sampling_size"])
                )
                var_expanded_ranges.append(list(var_range))
                var_names.append(arg)

    log.info(f"Required parameters {required_parameters}")
    log.info(f"Forbidden parameters {forbidden_parameters}")

    # Remove intersecting parameters from the forbidden list.
    for p in set(required_parameters) & set(forbidden_parameters):
        log.info(f"Intersecting parameter {p}")
        forbidden_parameters.remove(p)
    # Check if all the required parameters are specified.
    for p in required_parameters:
        if p not in args_dict.keys() or args_dict[p] is None:
            raise ValueError(f"Required parameter {p} not present.")
    # Check if none of the incompatible parameters are required.
    for p in forbidden_parameters:
        if p in args_dict.keys() and args_dict[p] is not None:
            raise ValueError(f"Forbidden parameter {p} is present")

    return var_names, var_expanded_ranges


def main(args):

    # Parse arguments provided to the parameter sweeper script.
    log.info("Parsing arguments...")

    args_dict = vars(args)

    # If any of the parameters related to the simulation are None,
    # set them to the default value provided in the configuration file.
    set_default(args_dict)

    # Check and expand the parameters in the provided ranges.
    var_names, var_expanded_ranges = check_expand_args(args_dict)

    if args_dict["sampling_type"] == "grid":
        # Create a generator of all the possible combinations of parameters based on
        # their expanded range lists
        parameter_sets_gen = itertools.product(*var_expanded_ranges)

    elif args_dict["sampling_type"] == "random":
        # Create a generator of the random sets of parameters.
        var_expanded_ranges = np.array(var_expanded_ranges).T.tolist()
        parameter_sets_gen = list(map(tuple, var_expanded_ranges))

    # Queue each set of parameter as a different simulation in the pool.
    log.info("Generating simulation parameter sets...")

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulation_arguments_path = pathlib.Path().joinpath(
        output_path, "simulation_arguments.txt"
    )

    simulation_number: int = 0

    with open(simulation_arguments_path, "w") as f_sa:
        for s in parameter_sets_gen:
            log.info(f"parameter set for simulation {simulation_number}:")
            log.info(s)

            # Generate output folder for the simulation.
            simulation_output_path = pathlib.Path().joinpath(
                output_path, f"{simulation_number:06}"
            )
            simulation_output_path.mkdir(parents=True, exist_ok=True)

            # Pack combination into a JSON override file and write it to the run
            # folder for this simulation.
            simulation_override_json = {}
            for i in range(len(s)):
                simulation_override_json[var_names[i]] = s[i]

            simulation_override_json_path = pathlib.Path().joinpath(
                simulation_output_path, "override.json"
            )

            with open(simulation_override_json_path, "w") as f:
                json.dump(
                    simulation_override_json, f, indent=4, sort_keys=True
                )

            f_sa.write(
                f"{simulation_output_path} {simulation_override_json_path}"
                + "\n"
            )

            simulation_number += 1

    log.info("Generating parameter sets completed!")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--output_dir",
        nargs="?",
        type=str,
        required=True,
        help="Path to the directory where the multi-run output is saved.",
    )

    args.add_argument(
        "--sampling_type",
        nargs="?",
        type=str,
        required=True,
        default="grid",
        help="Type of sampling for the parameter space of the simulation. Choose between grid or random.",
    )

    args.add_argument(
        "--sampling_size",
        nargs="?",
        type=int,
        default=None,
        help="Number of random values to draw for each simulation parameter. This parameter is required "
        "only if the sampling_type is set to random.",
    )

    args.add_argument(
        "--kick_model",
        nargs="?",
        type=str,
        default="km_exp",
        help="PDF model for the kick velocity. Choose between km_exp or km_maxwell.",
    )

    args.add_argument(
        "--sigma_k",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of kick velocity sigma for the Maxwell model with number of values "
        "[low, high, n_values]."
        "In random mode: range of kick velocity sigma for the Maxwell model [low, high].",
    )

    args.add_argument(
        "--vk_c",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of kick velocity vk_c for the exponential model with number of values "
        "[low, high, n_values]."
        "In random mode: range of kick velocity vk_c for the exponential model [low, high].",
    )

    args.add_argument(
        "--h_c",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of scale height h_c of the thin disk model with number of values "
        "[low, high, n_values]."
        "In random mode: range of scale height h_c of the thin disk model [low, high].",
    )

    args.add_argument(
        "--P_initial_mean",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of the mean initial spin period with number of values "
        "[low, high, n_values]."
        "In random mode: range of the mean initial spin period [low, high].",
    )

    args.add_argument(
        "--P_initial_sigma",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of the dispersion of the initial spin period with number of values "
        "[low, high, n_values]."
        "In random mode: range of the dispersion of the initial spin period [low, high].",
    )

    args.add_argument(
        "--B_initial_log10_mean",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range for the mean of the log10 initial magnetic field strength with number of values "
        "[low, high, n_values]."
        "In random mode: range of the mean of the log10 initial magnetic field strength [low, high].",
    )

    args.add_argument(
        "--B_initial_log10_sigma",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range for the dispersion of the log10 of the initial magnetic field strength "
        "with number of values [low, high, n_values]."
        "In random mode: range of the dispersion of the log10 initial magnetic field strength [low, high].",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
