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

import scripts.parameter_set_generator as psg

log = logging.getLogger(__name__)

unpaused = None
starting = None


def run_simulation(command: str) -> typing.Tuple[pathlib.Path, str]:
    """
    Run simulation command.
    This is the main routine for running a particular simulation. It runs the
    provided simulation command (a Python call to the simulation script with a
    set of CLI arguments) and captures all the output of the process.

    Args:
        command (List): full command to execute the simulation.

    Returns:
        The simulation command and the output of the process.
    """

    # Acquire the lock and block any other process from executing for two seconds.
    starting.acquire()
    threading.Timer(1, starting.release).start()

    # Once the process has released the lock for another process
    # it can proceed with the execution of the experiment.
    log.info(f"Launching simulation {command}")

    process_output = subprocess.check_output(
        command, stderr=subprocess.STDOUT, shell=True
    )

    log.info("Experiment finished...")

    return command, process_output.decode("utf-8")


def log_simulation(process_result: typing.Tuple[pathlib.Path, str]) -> None:

    """
    Callback to log all the info returned from a simulation run.

    Args:
        process_result: tuple containing the process simulation command and the
                        whole process output to console string.

    Returns:
        Nothing.
    """

    log.info("")
    log.info(
        "****************************************************************"
    )
    log.info(f"Ran simulation {process_result[0]}!")
    log.info(f"Process output:\n {process_result[1]}")
    log.info("Process finished...")


def setup_process_pool(event: mp.Event, lock: mp.Lock) -> None:

    """
    Set up the process pool for multiprocessing with a global pause/resume event.

    Args:
        event: reference to a master process event that will signal the child
               processes to pause or resume execution.
        lock: a reference to a master process lock that will coordinate the
              child process launching with waiting times.

    Returns:
        Nothing.
    """

    global unpaused
    unpaused = event

    global starting
    starting = lock


def main(args):
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

    # Check and expand the parameters in the provided ranges.
    var_names, var_expanded_ranges = psg.check_expand_args(args_dict)

    if args_dict["sampling_type"] == "grid":
        # Create a generator of all the possible combinations of parameters based on their expanded range lists
        parameter_sets_gen = itertools.product(*var_expanded_ranges)

    elif args_dict["sampling_type"] == "random":
        # Create a generator of the random sets of parameters.
        var_expanded_ranges = np.array(var_expanded_ranges).T.tolist()
        parameter_sets_gen = list(map(tuple, var_expanded_ranges))

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
        cmd: str = f"python examples/simulator/{simulator_type}.py"
        cmd += f" --output_dir {simulation_output_path}"
        cmd += f" --parameter_override {simulation_override_json_path}"
        if simulator_type == "simulate_population_magrot_det":
            cmd += f" --dyn_data {dyn_data_path}"

        pool.apply_async(run_simulation, args=(cmd,), callback=log_simulation)

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
