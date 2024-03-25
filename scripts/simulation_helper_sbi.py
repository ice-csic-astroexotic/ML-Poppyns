"""
    Simulator helper script.

    This script allows us to run the various simulator scripts in a multithreaded way.

    If the --sampling_type argument is set to "grid", we require the following for each tunable parameter:

        --argument low high count

    This expands the parameter to a linspace between [low, high] with a "count" number of steps.

    If the --sampling_type argument is set to "random", we require the following for each tunable parameter:

        --argument low high

    This expands the parameter to a list of values between [low, high] drawn from a uniform distribution.
    In this case, the number of values to be drawn for each parameter is specified by the argument --sampling_size.

    Both expansion types are evaluated for each specified argument. Subsequently, a generator produces all possible
    parameter combinations if in "grid" mode or sets of random parameter values if in "random" mode.
    Each parameter combination will spawn a new process that enters a multithreaded pool for later execution,
    allowing the asynchronous simulation of many populations in parallel with a defined maximum number of threads.

    NOTE: if an error occurs in one of the simulations, the script will not stop until all the processes will be
    terminated. The error will be only shown on the terminal in this case.

    Running the code:

        python3 simulator_helper.py --h
        To obtain help about all the arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)

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
import multiprocessing as mp
import pathlib
import sys

import torch
from torch.distributions import Distribution

from pypopsyn.learning.loaders.loader_multichannel_array_stat import (
    DatasetMultichannelArray,
)
from pypopsyn.simulator.configuration import cfg
from scripts.simulation_helper import (
    log_simulation,
    run_simulation,
    setup_process_pool,
)

log = logging.getLogger(__name__)


def simulator(
    args: argparse.Namespace,
    prior: Distribution,
    dataset: DatasetMultichannelArray,
) -> None:

    """
    Execute simulations based on the provided prior distribution.

    Args:
        args (argparse.Namespace): Arguments passed by command-line.
        prior (Distribution): Prior distribution.
        dataset (DatasetMultichannelArray): Dataset for simulations.
    """
    # Event on the master process that will be used to synchronize the child
    # processes and signal them for execution in the pool.
    event = mp.Event()

    # Lock on the master process to impose a delay in the process execution
    # so that none of them can be launched exactly at the same time.
    lock = mp.Lock()

    # A pool of processes with a defined capacity, a process spawning setup
    # routine and a general event to signal process execution.
    log.info(f"Initializing pool with {args.processes} processes...")

    pool = mp.Pool(
        args.processes,
        setup_process_pool,
        (
            event,
            lock,
        ),
    )

    # Parse arguments provided to the simulation helper script.
    log.info("Parsing arguments...")

    args_dict = vars(args)

    var_names = dataset.target_names
    # Save the statistics for the filtered labels.
    par_max = torch.tensor(dataset.target_max)
    par_min = torch.tensor(dataset.target_min)
    par_std = torch.tensor(dataset.target_std)
    par_mean = torch.tensor(dataset.target_mean)

    if args_dict["sampling_type"] == "prior":
        # Create a generator of the random sets of parameters using the prior distribution.
        parameter_sets_gen_tensor = prior.sample((args.sampling_size,))
        # If the parameters were normalized or standardized rescale quantities to their physical ranges.
        if dataset.normalize:
            parameter_sets_gen_tensor = (
                parameter_sets_gen_tensor * (par_max - par_min) + par_min
            )

        elif dataset.standardize:
            parameter_sets_gen_tensor = (
                parameter_sets_gen_tensor * par_std + par_mean
            )

        parameter_sets_gen = [
            tuple(subtensor.tolist())
            for subtensor in parameter_sets_gen_tensor
        ]
    else:
        raise ValueError(
            "The specified sampling type is not feasible, choose between grid or random."
        )

    # Set the simulation type and the path to the dynamical database if required.
    simulator_type = args_dict["simulator_type"]
    dyn_data_path = ""
    if simulator_type == "simulate_population_magrot_det":
        dyn_data_path = args_dict["dyn_data"]

    # Queue each set of parameter as a different simulation in the pool.
    log.info("Queuing simulations...")

    simulation_number: int = 0
    for s in parameter_sets_gen:
        log.info("Queuing simulation: ")
        log.info(s)

        # Generate output folder for the simulation.
        # Note that the numbering of the folders is limited to 6 digits here,
        # i.e., we can only generate simulations below 10 million.
        simulation_output_path = pathlib.Path().joinpath(
            args.output_dir, f"{simulation_number:06}"
        )
        simulation_output_path.mkdir(parents=True, exist_ok=True)

        # Pack combination into a JSON override file and write it to the folder for a given simulation.
        simulation_override_json = {}
        for i in range(len(s)):
            simulation_override_json[var_names[i]] = s[i]

        simulation_override_json_path = pathlib.Path().joinpath(
            simulation_output_path, "override.json"
        )

        with open(simulation_override_json_path, "w") as f:
            json.dump(simulation_override_json, f, indent=4, sort_keys=True)

        # Generate list for the command which consists of the python interpreter,
        # the script path and the path for the JSON override.
        server_path = cfg["path_server_output"]
        cmd: str = (
            f"python {server_path}examples/simulator/{simulator_type}.py"
        )
        cmd += f" --output_dir {simulation_output_path}"
        cmd += f" --parameter_override {simulation_override_json_path}"
        if simulator_type == "simulate_population_magrot_det":
            cmd += f" --dyn_data {dyn_data_path}"

        pool.apply_async(
            run_simulation,
            args=(cmd,),
            callback=log_simulation,
            error_callback=log.error,
        )

        simulation_number += 1

    log.info("")
    log.info("***************************************************************")
    log.info("Launching simulations")
    log.info("***************************************************************")

    # Signal the processes to begin execution in the pool.
    event.set()

    # Wait for all processes to finish.
    pool.close()
    pool.join()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyPopSyn parameters")

    args.add_argument(
        "--simulator_type",
        nargs="?",
        type=str,
        required=True,
        help="Name of the simulator script you want to run. Choose between simulate_population_full, "
        "simulate_population_dyn or simulate_population_magrot_det.",
    )

    args.add_argument(
        "--dyn_data",
        nargs="?",
        type=str,
        default=None,
        help="If using the simulator simulate_population_magrot_det, path to the file where "
        "the dynamically evolved population database is stored.",
    )

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
        "--processes",
        nargs="?",
        type=int,
        default=1,
        help="Number of simultaneous processes for the pool.",
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
        help="In grid mode: range of kick-velocity sigma for the Maxwell model with number of values "
        "[low, high, n_values]."
        "In random mode: range of kick-velocity sigma for the Maxwell model [low, high].",
    )

    args.add_argument(
        "--vk_c",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of kick-velocity vk_c for the exponential model with number of values "
        "[low, high, n_values]."
        "In random mode: range of kick-velocity vk_c for the exponential model [low, high].",
    )

    args.add_argument(
        "--h_c",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range of scale height h_c of the thin-disk model with number of values "
        "[low, high, n_values]."
        "In random mode: range of scale height h_c of the thin-disk model [low, high].",
    )

    args.add_argument(
        "--spin_period_model",
        nargs="?",
        type=str,
        default="log-normal",
        help="PDF model for the spin period. Choose between normal or log-normal.",
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

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    simulator(args)
