"""
    Parameter-set generator module.

    This module contains the methods necessary to produce a sweep of the simulation parameters,
    used in the parameter_sweeper.py and the simulation_helper.py script.

    If the --sampling_type argument is set to "grid", we require the following for each tunable parameter:

        --argument low high count

    This expands the parameter to a linspace between [low, high] with a "count" number of steps.

    If the --sampling_type argument is set to "random", we require the following for each tunable parameter:

        --argument low high

    This expands the parameter to a list of values between [low, high] drawn from a uniform distribution.
    In this case, the number of values to be drawn for each parameter is specified by the argument --sampling_size.

    Both expansion types are evaluated for each specified argument. Subsequently, a generator produces all possible
    parameter combinations if in "grid" mode or sets of random parameter values if in "random" mode.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2022

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
import pathlib
import sys

import numpy as np

from pypopsyn.simulator.configuration import cfg

log = logging.getLogger(__name__)


def set_default_parameter(args_dict: dict, parameter_name: str) -> None:
    """
    If a parameter related to the simulation is None, set it to the
    default value provided in the configuration file.

    Args:
        args_dict (dict): dictionary of the parsed argument via CLI.
        parameter_name (str): name of the parameter to set.

    Returns:
        Nothing.
    """

    # For unspecified parameters, assign corresponding default value according to the chosen sampling approach.
    # We specify a list for compatibility reasons according to [default value, default value (, 1)], respectively.
    if args_dict[parameter_name] is None:
        if args_dict["sampling_type"] == "grid":
            args_dict[parameter_name] = [
                cfg[parameter_name],
                cfg[parameter_name],
                1,
            ]
        elif args_dict["sampling_type"] == "random":
            args_dict[parameter_name] = [
                cfg[parameter_name],
                cfg[parameter_name],
            ]

        log.info(
            "{} set to the default value {}".format(
                parameter_name, cfg[parameter_name]
            )
        )


def set_default_if_none(args_dict: dict) -> None:
    """
    If any of the parameters related to the simulation are None, set them to the
    default value provided in the configuration file.

    Args:
        args_dict (dict): dictionary of the parsed argument via CLI.

    Returns:
        Nothing.
    """

    # Setting the default parameters for the dynamical evolution.
    if args_dict["kick_model"] == "km_maxwell":
        set_default_parameter(args_dict, "sigma_k")

    if args_dict["kick_model"] == "km_exp":
        set_default_parameter(args_dict, "vk_c")

    set_default_parameter(args_dict, "h_c")

    # Setting the default parameters for the magneto-rotational evolution.
    if args_dict["spin_period_model"] == "normal":
        set_default_parameter(args_dict, "P_initial_mean")
        set_default_parameter(args_dict, "P_initial_sigma")

    if args_dict["spin_period_model"] == "log-normal":
        set_default_parameter(args_dict, "P_initial_log10_mean")
        set_default_parameter(args_dict, "P_initial_log10_sigma")

    set_default_parameter(args_dict, "B_initial_log10_mean")
    set_default_parameter(args_dict, "B_initial_log10_sigma")
    set_default_parameter(args_dict, "a_late")


def expand_parameter(
    args_dict: dict, parameter_name: str, range_values: list
) -> np.ndarray:
    """
    Expand the simulation parameters.
    If in grid mode: expand each simulation parameter in linear space in the specified ranges.
    If in random mode: draw random set of parameter values from uniform distributions in the specified ranges.

    Args:
        args_dict (dict): dictionary of the parsed argument via CLI.
        parameter_name (str): name of the parameter to expand.
        range_values (list): range of values where to expand the parameter.

    Returns:
        (np.ndarray): a list containing the expanded range of the parameter.
    """

    expanded_parameter = []

    if args_dict["sampling_type"] == "grid":
        # The three values [low, high, steps] are used to expand each of the arguments
        # with linear spacing in the range [low, high] with a number of specified steps.
        if len(range_values) != 3:
            raise ValueError(
                f"In grid mode the list must have length 3 for parameter {parameter_name}"
            )
        expanded_parameter = np.linspace(
            range_values[0], range_values[1], int(range_values[2])
        )

    elif args_dict["sampling_type"] == "random":
        # The two values [low, high] define the range from which a number of values
        # (specified according to the sampling_size argument) is drawn from a uniform distribution.
        if len(range_values) != 2:
            raise ValueError(
                f"In random mode the list must have length 2 for parameter {parameter_name}"
            )
        expanded_parameter = np.random.uniform(
            range_values[0], range_values[1], int(args_dict["sampling_size"])
        )

    return expanded_parameter


def check_expand_args(args_dict: dict) -> (list, list):
    """
    Check if the parsed input arguments are coherent and have the correct shape.
    If in grid mode: expand each simulation parameter in linear space in the specified ranges.
    If in random mode: draw random set of parameter values from uniform distributions in the specified ranges.

    Args:
        args_dict (dict): dictionary of the parsed argument via CLI.

    Returns:
        (list, list): a list containing the expanded ranges of the parameters and a list containing the names of the
        expanded parameters.
    """

    # If any of the parameters related to the simulation are None,
    # set them to the default value provided in the configuration file.
    set_default_if_none(args_dict)

    cli_args: list = []
    cli_str: list = []

    path_server_software = cfg["path_server_software"]
    # Open the parameter dictionary to load requirements.
    config_sweeper_path = pathlib.Path().joinpath(
        path_server_software,
        "scripts/config_sweeper.json",
    )
    f = open(config_sweeper_path)
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
            # the directory path where the multirun output is saved, the directory path where
            # a dynamical database is saved or a selection parameter.
            # In the latter case we have to: (a) check whether the selection is valid,
            # (b) capture the list of required parameters, and (c) gather the forbidden ones
            # (i.e., those that belong to other types of selections).
            if arg == "output_dir":
                cli_args.append("--output_dir")
                cli_str.append(value)
            elif arg == "dyn_data":
                cli_args.append("--dyn_data")
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
                # If the value for an argument is not in the dictionary of
                # possible values, we throw an exception.
                raise ValueError(
                    f"The value {value} is not feasible for parameter {arg}"
                )

        elif type(value) is list:
            # If the value is a list, we assume it will be a specification of three values
            # if sampling_type = grid or two values if sampling_type = random.
            # We then expand the parameter accordingly.
            var_range = expand_parameter(args_dict, arg, value)
            var_expanded_ranges.append(list(var_range))
            var_names.append(arg)

    log.info(f"Required parameters {required_parameters}")
    log.info(f"Forbidden parameters {forbidden_parameters}")

    # Check if all the required parameters are specified.
    for p in required_parameters:
        if p not in args_dict.keys() or args_dict[p] is None:
            raise ValueError(f"Required parameter {p} not present.")

    # Check if none of the incompatible parameters are required.
    for p in forbidden_parameters:
        if p in args_dict.keys() and args_dict[p] is not None:
            raise ValueError(f"Forbidden parameter {p} is present")

    return var_names, var_expanded_ranges
