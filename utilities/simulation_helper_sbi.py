"""
    Simulator helper script.

    This script allows us to run various simulator scripts in a multithreaded manner. Unlike the `simulation_helper.py`
    script, which sample parameters randomly or on a grid, this script follows a prior distribution for parameter
    sampling. Note that this script can be run only when using a prior distribution from the sbi package that has the
    .sample() method available.

    The number of values to be drawn for each parameter is specified by the argument --sampling_size.

    Each parameter combination will spawn a new process when the `simulator_multiprocess` function is called.
    These processes enter a multithreaded pool for later execution, allowing the asynchronous simulation of many
    populations in parallel with a defined maximum number of threads. However, when the `simulator_dask` function is
    called, the multithreading is handled with Dask, enabling parallel execution of simulations across the Dask cluster
    in HTCondor.

    NOTE: if an error occurs in one of the simulations, the script will not stop until all the processes have been
    terminated. The error will be only shown in the terminal in this case.

    Running the code:

        python simulator_helper_sbi.py --h
        To obtain help about all the arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
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

import json
import logging
import multiprocessing as mp
import pathlib
from logging import Logger

import dask
import torch
from dask.distributed import Client
from dask_jobqueue import HTCondorCluster
from sbi.inference.posteriors.direct_posterior import DirectPosterior

import pypopsyn.learning.configuration_parser as configuration_parser
from pypopsyn.learning.loaders.loader_multichannel_array_stat import (
    DatasetMultichannelArray,
)
from pypopsyn.simulator.config_simulator import cfg
from utilities.simulation_helper import (
    log_simulation,
    run_simulation,
    run_simulation_dask,
    setup_process_pool,
)

log = logging.getLogger(__name__)


def initialize_dask_cluster(
    logger: Logger, config: configuration_parser.ConfigurationParser
) -> HTCondorCluster:
    """
    Initialize a Dask cluster for distributed computing.

    Args:
        logger (Logger): Logger object.
        config (ConfigurationParser): Configuration object specifying dataset loading parameters.

    Returns:
        HTCondorCluster: Initialized Dask cluster object.
    """

    # Creating a folder to save the stdout and stderr of the terminal for each worker.
    htcondor_output_folder = f"{config.save_dir}/htcondor_output"
    pathlib.Path(htcondor_output_folder).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Saving the stdout and stderr of the terminal of each worker in {htcondor_output_folder}."
    )
    # Creating the cluster with dask for HTCondor.
    extra = {
        "getenv": "True",
        "output": f"{htcondor_output_folder}/$(ClusterId)_$(ProcId)-out.txt",
        "error": f"{htcondor_output_folder}/$(ClusterId)_$(ProcId)-err.txt",
        "+flavour": "long",
    }

    # Specifying computing requirements as needed for a single magneto-thermal simulation.
    cluster = HTCondorCluster(
        cores=1, memory="2 GB", disk="2 GB", job_extra_directives=extra
    )

    # Scaling the cluster to the number of workers specified in the configuration file.
    num_workers_dask = config["workers_dask"]
    cluster.scale(num_workers_dask)

    # Wait for at least one worker to be ready.
    cluster.wait_for_workers(1)

    # Create Dask client connected to the cluster.
    client = Client(cluster)

    # Start the Dask dashboard for monitoring.
    logger.info(f"Dask client {client.dashboard_link}")

    return cluster


def simulator_dask(
    args_dict: dict,
    prior: DirectPosterior,
    dataset: DatasetMultichannelArray,
    device: torch.device,
) -> None:

    """
    Execute simulations based on the provided prior distribution in a multithreaded manner using the `Dask` package.

    Args:
        args_dict (Dictionary): Dictionary with the arguments.
        prior (DirectPosterior): Prior distribution.
        dataset (DatasetMultichannelArray): Stores statistics and scaling information used in the prior distribution.
        device (torch.device): Device used to run the script.

    Returns:
        None
    """
    # Create a list to hold delayed computations for each simulation.
    delayed_simulations = []

    # Parse arguments provided to the simulation helper script.
    log.info("Parsing arguments...")

    # Extracting the names of the parameters.
    var_names = dataset.target_names

    # Save the statistics for the filtered labels.
    par_max = torch.tensor(dataset.target_max).to(device)
    par_min = torch.tensor(dataset.target_min).to(device)
    par_std = torch.tensor(dataset.target_std).to(device)
    par_mean = torch.tensor(dataset.target_mean).to(device)

    # Create a generator of the random sets of parameters using the prior distribution.
    parameter_sets_gen_tensor = prior.sample((args_dict["sampling_size"],))

    # If the parameters were normalized or standardized, rescale quantities to their physical ranges.
    if dataset.normalize:
        parameter_sets_gen_tensor = (
            parameter_sets_gen_tensor * (par_max - par_min) + par_min
        )

    elif dataset.standardize:
        parameter_sets_gen_tensor = (
            parameter_sets_gen_tensor * par_std + par_mean
        )

    parameter_sets_gen = [
        tuple(subtensor.tolist()) for subtensor in parameter_sets_gen_tensor
    ]

    # Set the simulation type and the path to the dynamical database if required.
    simulator_type = args_dict["simulator_type"]
    dyn_data_path = ""

    if simulator_type == "simulate_population_magrot_det":
        dyn_data_path = args_dict["dyn_data"]

    # Queue each set of parameters as a different simulation in the pool.
    log.info("Queuing simulations...")

    simulation_number: int = 0

    # Create a delayed version of the 'run_simulation_dask' function using Dask that allows for lazy evaluation.
    # This enables parallel processing capabilities within Dask.
    run_simulation_delayed = dask.delayed(run_simulation_dask)

    for s in parameter_sets_gen:
        log.info("Queuing simulation: ")
        log.info(s)

        # Generate output folder for the simulation.
        # Note that the numbering of the folders is limited to 6 digits here,
        # i.e., we can only generate simulations below 10 million.
        simulation_output_path = pathlib.Path().joinpath(
            args_dict["output_dir"], f"{simulation_number:06}"
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

        # Generate a list for the command (cmd), including the Python interpreter, the script path specified with
        # 'simulator_type', and the path for the JSON override.
        server_path = cfg["path_to_software"]
        cmd: str = f"python {server_path}pyposyn/simulator/{simulator_type}.py"
        cmd += f" --output_dir {simulation_output_path}"
        cmd += f" --parameter_override {simulation_override_json_path}"
        if simulator_type == "simulate_population_magrot_det":
            cmd += f" --dyn_data {dyn_data_path}"

        # Create delayed computation for each simulation.
        delayed_simulations.append(run_simulation_delayed(cmd))

        simulation_number += 1

    log.info("")
    log.info("***************************************************************")
    log.info("Launching simulations")
    log.info("***************************************************************")

    # Compute the delayed computations, i.e., run the simulations in parallel with HTCondor.
    dask.compute(delayed_simulations)


def simulator_multiprocess(
    args_dict: dict,
    prior: DirectPosterior,
    dataset: DatasetMultichannelArray,
) -> None:
    """
    Execute simulations based on the provided prior distribution in a multithreaded manner using the `multiprocessing`
    package.

    Args:
        args_dict (Dictionary): Dictionary with the arguments.
        prior (DirectPosterior): Prior distribution.
        dataset (DatasetMultichannelArray):  Stores statistics and scaling information used in the prior distribution.

    Returns:
        None
    """
    # Event on the master process that will be used to synchronize the child
    # processes and signal them for execution in the pool.
    event = mp.Event()

    # Lock on the master process to impose a delay in the process execution
    # so that none of them can be launched exactly at the same time.
    lock = mp.Lock()

    # A pool of processes with a defined capacity, a process spawning setup
    # routine and a general event to signal process execution.
    nprocesses = args_dict["processes"]
    log.info(f"Initializing pool with {nprocesses} processes...")

    pool = mp.Pool(
        nprocesses,
        setup_process_pool,
        (
            event,
            lock,
        ),
    )

    # Parse arguments provided to the simulation helper script.
    log.info("Parsing arguments...")

    # Extracting the names of the parameters.
    var_names = dataset.target_names

    # Save the statistics for the filtered labels.
    par_max = torch.tensor(dataset.target_max)
    par_min = torch.tensor(dataset.target_min)
    par_std = torch.tensor(dataset.target_std)
    par_mean = torch.tensor(dataset.target_mean)

    # Create a generator of the random sets of parameters using the prior distribution.
    parameter_sets_gen_tensor = prior.sample((args_dict["sampling_size"],))

    # If the parameters were normalized or standardized, rescale quantities to their physical ranges.
    if dataset.normalize:
        parameter_sets_gen_tensor = (
            parameter_sets_gen_tensor * (par_max - par_min) + par_min
        )

    elif dataset.standardize:
        parameter_sets_gen_tensor = (
            parameter_sets_gen_tensor * par_std + par_mean
        )

    parameter_sets_gen = [
        tuple(subtensor.tolist()) for subtensor in parameter_sets_gen_tensor
    ]

    # Set the simulation type and the path to the dynamical database if required.
    simulator_type = args_dict["simulator_type"]
    dyn_data_path = ""
    if simulator_type == "simulate_population_magrot_det":
        dyn_data_path = args_dict["dyn_data"]

    # Queue each set of parameters as a different simulation in the pool.
    log.info("Queuing simulations...")

    simulation_number: int = 0
    for s in parameter_sets_gen:
        log.info("Queuing simulation: ")
        log.info(s)

        # Generate output folder for the simulation.
        # Note that the numbering of the folders is limited to 6 digits here,
        # i.e., we can only generate simulations below 10 million.
        simulation_output_path = pathlib.Path().joinpath(
            args_dict["output_dir"], f"{simulation_number:06}"
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

        # Generate a list for the command (cmd), including the Python interpreter, the script path specified with
        # 'simulator_type', and the path for the JSON override.
        server_path = cfg["path_to_software"]
        cmd: str = (
            f"python {server_path}pypopsyn/simulator/{simulator_type}.py"
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
