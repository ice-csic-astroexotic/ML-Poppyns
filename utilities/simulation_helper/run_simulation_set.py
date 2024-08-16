"""
    Simulator helper script.

    This script allows us to run the various simulator utilities in a multithreaded way.

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

    Display help message to run the code:

    python run_simulation_set.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import threading
import time
import typing
from typing import Any

import numpy as np

import pypopsyn.simulator.simulate_population_dyn as dyn
import pypopsyn.simulator.simulate_population_magrot_det as magrot
import utilities.simulation_helper.parameter_set_generator as psg
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)

unpaused = None
starting = None


def log_error(e: Exception) -> None:
    """
    Log an exception raised during the simulation process.

    Args:
        e (Exception): The exception to log.
    """
    log.error("An error occurred during the simulation.", exc_info=e)


def safe_copytree(
    src: str, dst: str, max_attempts: int = 3, delay: int = 5
) -> None:
    """
    Safely copy a directory tree with retries. This function attempts to copy a directory tree from the source path to
    the destination path. If the copy operation fails (e.g., due to connection issues), it will retry the operation
    a specified number of times with a delay between each attempt.

    Args:
        src (str): Source directory path.
        dst (str): Destination directory path.
        max_attempts (int): Maximum number of retry attempts. Default is 3 retries.
        delay (int): Delay between retry attempts in seconds. Default is 5 seconds.
    """
    attempts = 0

    while attempts < max_attempts:
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            # If copytree finishes successfully, exit function.
            return
        except Exception as e:
            # If an error occurs during the copy operation, log the error message,including the current time and the
            # machine name.
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            machine_name = platform.node()
            log.error(
                f"Attempt {attempts + 1} failed with error at {current_time} on {machine_name}: {e}"
            )
            # Wait for the seconds defined in the delay variable before retrying.
            time.sleep(delay)
            attempts += 1
            if attempts == max_attempts:
                # If the number of attempts reaches the maximum number, raise an exception.
                log.error("Maximum retry attempts reached, failing task.")
                raise


def robust_run_simulation_dask(
    *args: Any, max_attempts: int = 3, delay: int = 5, **kwargs: Any
) -> None:
    """
    This function wraps around the run_simulation_dask function, adding retry logic to handle transient issues
    (e.g., connection problems). If an error occurs during the run_simulation_dask function, it will retry the operation
    a specified number of times with a delay between each attempt.

    Args:
        args (Any): Variable length argument list.
        max_attempts (int, optional): Maximum number of retry attempts. Default is 3.
        delay (int, optional): Delay between retry attempts in seconds. Default is 5.
        kwargs (Any): Arbitrary keyword arguments.
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            run_simulation_dask(*args, **kwargs)
            # If run_simulation_dask finishes successfully, exit function.
            return
        except Exception as e:
            # If an error occurs during the run_simulation_dask function, log the error message, including the current
            # time and the machine name.
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            machine_name = platform.node()
            log.error(
                f"Attempt {attempts + 1} failed with error at {current_time} on {machine_name}: {e}"
            )
            # Wait for the seconds defined in the delay variable before retrying.
            time.sleep(delay)
            attempts += 1
            if attempts == max_attempts:
                # If the number of attempts reaches the maximum number raise an exception.
                log.error("Maximum retry attempts reached, failing task.")
                raise


def run_simulation_dask(
    args: argparse.Namespace,
    simulator_type: str,
    simulation_output_path: str,
    simulation_override_json: dict,
    dyn_data_path: str,
) -> None:
    """
    Run the simulation command, copying the output folder to the node before execution and back to the original location
    afterward to prevent overload at PIC. Unlike the run_simulation function below, this function does not capture all the
    terminal output of the process. This function is necessary for running `train_tsnpe.py` using Dask and HTCondor.

    Args:
        args (argparse.Namespace): Arguments required for the simulation, including output directory, parameter overrides,
            and optional dynamic data path.
        simulator_type (str): The type of simulator to use, determining the specific simulation script to run.
        simulation_output_path (str): Path to the simulation output folder.
        simulation_override_json (dict): Dictionary with the parameter values for the override.json file.
        dyn_data_path (str): Dynamical database path.
    """

    # Copy the dynamical database and the repository to the node if it is not already there.
    # This action prevents overloading the PIC with too many calls.
    try:
        if not os.path.exists(os.path.basename(dyn_data_path)):
            safe_copytree(
                dyn_data_path,
                os.path.basename(dyn_data_path),
            )
        if not os.path.exists("MAGNESIA_population_synthesis"):
            safe_copytree(
                "/data/magnesia/software/MAGNESIA_population_synthesis",
                "MAGNESIA_population_synthesis",
            )
        # Generate the output folder with the parameter_override.json file in each node.
        output_dir_path = pathlib.Path(args.save_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        with open(args.parameter_override, "w") as f:
            json.dump(simulation_override_json, f, indent=4, sort_keys=True)

        # Call either the simulate_population_magrot or simulate_population_dyn module depending on the case.
        if simulator_type == "simulate_population_magrot_det":
            magrot.simulate_population(args)
        else:
            dyn.simulate_population(args)

        # Copy the output folder back to the original location.
        safe_copytree(output_dir_path, simulation_output_path)
        # Remove the folder to prevent issues with overwriting.
        shutil.rmtree(output_dir_path)

        log.info(
            f"Copied output folder back to original location: {simulation_output_path}"
        )

    except subprocess.CalledProcessError as e:
        # Log any errors raised during the simulation.
        log.error(f"Error executing simulation with args: {args}")
        log.error(f"Error details: {str(e)}")
        raise

    log.info("Simulation finished")


def run_simulation(command: str) -> typing.Tuple[pathlib.Path, str]:
    """
    Run simulation command.

    This is the main routine for running a particular simulation. It runs the
    provided simulation command (a Python call to the simulation script with a
    set of CLI arguments) and captures all the output of the process.

    Args:
        command (str): Full command to execute the simulation.

    Returns:
        (Tuple[pathlib.Path, str]): The simulation command and the output of the process.
    """

    # Acquire the lock and block any other process from executing for two seconds.
    starting.acquire()
    threading.Timer(1, starting.release).start()

    # Once the process has released the lock for another process
    # it can proceed with the execution of the experiment.
    log.info(f"Launching simulation {command}")
    try:
        process_output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True
        )

        log.info("Experiment finished...")
        return command, process_output.decode("utf-8")

    except subprocess.CalledProcessError as e:
        log.error(f"Simulation failed with error code {e.returncode}")
        log.error(e.output.decode("utf-8"))
        return command, e.output.decode("utf-8")


def log_simulation(process_result: typing.Tuple[pathlib.Path, str]) -> None:
    """
    Callback to log all the info returned from a simulation run.

    Args:
        process_result (Tuple[pathlib.Path, str]): Tuple containing the process simulation command and the
            whole process output to console string.
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
        event (mp.Event): Reference to a master process event that will signal the child
            processes to pause or resume execution.
        lock (mp.Lock): A reference to a master process lock that will coordinate the
            child process launching with waiting times.
    """

    global unpaused
    unpaused = event

    global starting
    starting = lock


def main(args) -> None:
    """
    Execute parameterized simulations using a multiprocessing pool.

    This function initializes a multiprocessing pool to run simulations based on a set of parameters
    defined by the user. It parses the command-line arguments, expands parameter ranges for sampling,
    queues the simulations, and manages their execution in parallel. It also generates JSON files for
    each simulation that contain the parameters used for that run.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse. Required parameters include:
            - simulator_type (str): The name of the simulator script to run. Options include
              'simulate_population_full', 'simulate_population_dyn', or 'simulate_population_magrot_det'.
            - dyn_data (str): (Optional) Path to the dynamically evolved population database, required
              if using 'simulate_population_magrot_det'.
            - save_dir (str): Directory path where the output of the simulations will be saved.
            - sampling_type (str): Method for sampling the parameter space. Choose between 'grid' and 'random'.
            - sampling_size (int): Number of random values to draw for each simulation parameter (required
              if sampling_type is 'random').
            - processes (int): Number of simultaneous processes for the multiprocessing pool (default is 1).
            - sigma_k, vk_c, h_c, P_initial_mean, P_initial_sigma, P_initial_log10_mean,
              P_initial_log10_sigma, B_initial_log10_mean, B_initial_log10_sigma, a_late (list of float):
              Each of these parameters can accept a range defined for grid or random sampling as follows:
              - In grid mode: range specified as [low, high, n_values].
              - In random mode: range specified as [low, high].
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

    # Check and expand the parameters in the provided ranges.
    var_names, var_expanded_ranges = psg.check_expand_args(args_dict)

    if args_dict["sampling_type"] == "grid":
        # Create a generator of all the possible combinations of parameters based on their expanded range lists.
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
            args.save_dir, f"{simulation_number:06}"
        )
        simulation_output_path.mkdir(parents=True, exist_ok=True)

        # Save the set of parameter values into a JSON override file and write it to the folder for a given simulation.
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
        server_path = cfg["path_to_software"]
        cmd: str = (
            f"python {server_path}/pypopsyn/simulator/{simulator_type}.py"
        )
        cmd += f" --save_dir {simulation_output_path}"
        cmd += f" --parameter_override {simulation_override_json_path}"
        if simulator_type == "simulate_population_magrot_det":
            cmd += f" --dyn_data {dyn_data_path}"

        pool.apply_async(
            run_simulation,
            args=(cmd,),
            callback=log_simulation,
            error_callback=log_error,
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
        "--save_dir",
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
    args.add_argument(
        "--L_radio_log10_mean",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range for the mean of the log-normally distributed radio luminosity normalization factor."
        "with number of values [low, high, n_values]."
        "In random mode: range of the mean of the log-normally distributed radio luminosity normalization "
        "factor [low, high].",
    )
    args.add_argument(
        "--epsilon_L",
        nargs="*",
        type=float,
        default=None,
        help="In grid mode: range for the power-law slope of the intrinsic luminosity "
        "with number of values [low, high, n_values]."
        "In random mode: range of the power-law slope of the intrinsic luminosity [low, high].",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
