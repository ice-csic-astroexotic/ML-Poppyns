"""
    Parameter-sweeper script.

    This script generates the files necessary to launch multiple simulations with different parameter values
    using HTCondor at the PIC. It uses methods from the module parameter_set_generator.py.

    If the --sampling_type argument is set to "grid", we require the following for each tunable parameter:

        --argument low high count

    This expands the parameter to a linspace between [low, high] with a "count" number of steps.

    If the --sampling_type argument is set to "random", we require the following for each tunable parameter:

        --argument low high

    This expands the parameter to a list of values between [low, high] drawn from a uniform distribution.
    In this case, the number of values to be drawn for each parameter is specified by the argument --sampling_size.

    Both expansion types are evaluated for each specified argument. Subsequently, a generator produces all possible
    parameter combinations if in "grid" mode or sets of random parameter values if in "random" mode.
    Each set will be saved in a JSON "parameter_override" file that will be used as input to a simulation.

    Running the code:

        python3 parameter_sweeper.py --h
        To obtain help about all the arguments that can be used.

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

import scripts.parameter_set_generator as psg
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)


def main(args):
    # Parse arguments provided to the parameter-sweeper script.
    log.info("Parsing arguments...")

    args_dict = vars(args)

    # Check and expand the parameters in the provided ranges.
    var_names, var_expanded_ranges = psg.check_expand_args(args_dict)

    if args_dict["sampling_type"] == "grid":
        # Create a generator of all possible combinations of parameters based on their expanded range lists.
        parameter_sets_gen = itertools.product(*var_expanded_ranges)

    elif args_dict["sampling_type"] == "random":
        # Create a generator of the random sets of parameters.
        list_var_expanded_ranges = np.array(var_expanded_ranges).T.tolist()
        parameter_sets_gen = list(map(tuple, list_var_expanded_ranges))

    else:
        raise ValueError(
            "The specified sampling type is not feasible, choose between grid or random."
        )

    # Save the input arguments to run each simulation in a file.
    log.info("Generating simulation parameter sets...")

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulation_arguments_path = pathlib.Path().joinpath(
        output_path, "simulation_arguments.txt"
    )

    simulation_number: int = 0

    with open(simulation_arguments_path, "w") as f_sa:
        for s in parameter_sets_gen:
            log.info(f"Parameter set for simulation {simulation_number}:")
            log.info(s)

            # Generate output folders for the simulations.
            # Note that the numbering of the folders is limited to 6 digits here,
            # i.e., we can only generate simulations below 10 million.
            path_to_output = cfg["path_to_output"]
            output_path = pathlib.Path().joinpath(path_to_output, output_path)
            simulation_output_path = pathlib.Path().joinpath(
                output_path, f"{simulation_number:06}"
            )
            simulation_output_path.mkdir(parents=True, exist_ok=True)

            # Pack combination into a JSON override file and write it to the folder for a given simulation.
            simulation_override_json = {}
            for i in range(len(s)):
                simulation_override_json[var_names[i]] = s[i]

            simulation_output_path = pathlib.Path().joinpath(
                path_to_output, simulation_output_path
            )
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
        "--P_initial_log10_mean",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of the log10 mean initial spin period with number of values "
        "[low, high, n_values]."
        "In random mode: range of the log10 mean initial spin period [low, high].",
    )

    args.add_argument(
        "--P_initial_log10_sigma",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of the log10 dispersion of the initial spin period with number of values "
        "[low, high, n_values]."
        "In random mode: range of the log10 dispersion of the initial spin period [low, high].",
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

    args.add_argument(
        "--a_late",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range for the power-law slope of the late time magnetic field evolution "
        "with number of values [low, high, n_values]."
        "In random mode: range of the power-law slope of the late time magnetic field evolution [low, high].",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
