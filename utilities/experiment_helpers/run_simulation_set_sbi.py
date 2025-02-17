"""
    Simulator helper script.

    This script allows us to run various simulator scripts in a multithreaded manner. Unlike the `run_simulation_set.py`
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

    Display help message to run the code:

    python run_simulation_set_sbi.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
from logging import Logger
from typing import Any

import dask
import torch
from dask.distributed import Client
from dask_jobqueue import HTCondorCluster
from sbi.inference.posteriors.direct_posterior import DirectPosterior

import pypopsyn.learning.configuration_parser as configuration_parser
from pypopsyn.learning.loaders.loader_multichannel_array import (
    DatasetMultichannelArray,
)
from pypopsyn.simulator.config_simulator import cfg
from utilities.experiment_helpers.run_simulation_set import (
    log_simulation,
    robust_run_simulation_dask,
    run_simulation,
    setup_process_pool,
)

log = logging.getLogger(__name__)

# Forcing Dask to wait 120 s before considering an unresponsive worker as dead.
dask.config.set({"distributed.comm.timeouts.tcp": "120s"})


def sample_without_nan(
    distribution: Any,
    sampling_size: int,
    device: torch.device,
    max_attempts: int = 20,
) -> torch.Tensor:
    """
    Sample a distribution while removing NaN values from the sampled outputs.
    Stops after max_attempts if sufficient valid samples are not obtained.

    Args:
        distribution (Any): The distribution to sample from.
        sampling_size (int): The number of samples to draw from the distribution.
        device (torch.device): Device used to run the script.
        max_attempts (int): The maximum number of attempts to sample (default is 20).

    Returns:
        torch.Tensor: A tensor of samples, with shape [sampling_size, D] (where D is the dimensionality of the samples).
                      All NaN values have been removed. If not enough valid samples are obtained, returns fewer samples.
    """

    samples = []
    attempts = 0

    while len(samples) < sampling_size and attempts < max_attempts:
        remaining_samples = sampling_size - len(samples)

        new_samples = distribution.sample(
            (remaining_samples,), show_progress_bars=False
        )

        valid_samples = new_samples[
            ~torch.any(torch.isnan(new_samples), dim=1)
        ]

        samples.extend(valid_samples.tolist())

        attempts += 1

    if len(samples) < sampling_size:
        raise RuntimeError(
            f"Unable to obtain {sampling_size} valid samples after {max_attempts} attempts."
        )

    return torch.tensor(samples).to(device)


def initialize_dask_cluster(
    logger: Logger, config: configuration_parser.ConfigurationParser
) -> HTCondorCluster:
    """
    Initialize a Dask cluster for distributed computing.

    Args:
        logger (Logger): Logger object.
        config (ConfigurationParser): Configuration object specifying dataset loading parameters.

    Returns:
        (HTCondorCluster): Initialized Dask cluster object.
    """

    # Creating a folder to save the stdout and stderr of the terminal for each worker.
    htcondor_output_folder = f"{config.save_dir}/htcondor_output"
    pathlib.Path(htcondor_output_folder).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Saving the stdout and stderr of the terminal for each worker in {htcondor_output_folder}."
    )
    # Creating the cluster with Dask for HTCondor.
    extra = {
        "getenv": "True",
        "output": f"{htcondor_output_folder}/$(ClusterId)_$(ProcId)-out.txt",
        "error": f"{htcondor_output_folder}/$(ClusterId)_$(ProcId)-err.txt",
        "+flavour": '"long"',
    }

    # Specifying the computing requirements as needed for a single magneto-thermal simulation.
    # If nanny is set to True, each worker is started by a nanny process which can restart the worker if it fails.
    # We set 1 thread per worker to prevent system overload. The timeout duration to wait for a worker to start is set
    # to 60 seconds.

    cluster = HTCondorCluster(
        cores=1,
        memory="2 GB",
        disk="2 GB",
        job_extra_directives=extra,
        nanny=True,
        death_timeout="60s",
        worker_extra_args=["--nthreads", "1"],
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
    Execute simulations based on the provided prior distribution in a multithreaded manner using the Dask package.

    Args:
        args_dict (Dictionary): Dictionary with the arguments.
        prior (DirectPosterior): Prior distribution.
        dataset (DatasetMultichannelArray): Stores statistics and scaling information used in the prior distribution.
        device (torch.device): Device used to run the script.
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
    parameter_sets_gen_tensor = sample_without_nan(
        prior, args_dict["sampling_size"], device
    )

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
    run_simulation_delayed = dask.delayed(robust_run_simulation_dask)

    for s in parameter_sets_gen:
        log.info("Queuing simulation: ")
        log.info(s)

        # Setting output folder path for each simulation.
        # Note that the numbering of the folders is limited to 6 digits here,
        # i.e., we can only generate simulations below 10 million.
        folder_name = f"{simulation_number:06}"
        simulation_output_path_original = pathlib.Path().joinpath(
            args_dict["save_dir"], folder_name
        )
        # Save the set of parameter values into a dictionary.
        simulation_override_json = {}
        for i in range(len(s)):
            simulation_override_json[var_names[i]] = s[i]

        simulation_override_json_path = pathlib.Path().joinpath(
            folder_name, "override.json"
        )
        # Generate a list for the command (cmd), including the Python interpreter, the script path specified with
        # 'simulator_type', and the path for the JSON override.
        # Prepare arguments for the simulate_population function.
        simulation_args = argparse.Namespace(
            save_dir=folder_name,
            parameter_override=simulation_override_json_path,
            dyn_data=os.path.basename(dyn_data_path),
        )

        # Create delayed computation for each simulation. Each simulation will be attempted up to 5 times in case of an
        # error, with a 10-second wait between each attempt. This is done to avoid stopping the entire training process
        # if there is a connection issue with a worker.

        delayed_simulations.append(
            run_simulation_delayed(
                simulation_args,
                simulator_type,
                simulation_output_path_original,
                simulation_override_json,
                dyn_data_path,
                max_attempts=5,
                delay=10,
            )
        )

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
    device: torch.device,
) -> None:
    """
    Execute simulations based on the provided prior distribution in a multithreaded manner using the package
    multiprocessing.

    Args:
        args_dict (Dictionary): Dictionary with the arguments.
        prior (DirectPosterior): Prior distribution.
        dataset (DatasetMultichannelArray):  Stores statistics and scaling information used in the prior distribution.
        device (torch.device): Device used to run the script.
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
    par_max = torch.tensor(dataset.target_max).to(device)
    par_min = torch.tensor(dataset.target_min).to(device)
    par_std = torch.tensor(dataset.target_std).to(device)
    par_mean = torch.tensor(dataset.target_mean).to(device)

    # Create a generator of the random sets of parameters using the prior distribution.
    parameter_sets_gen_tensor = sample_without_nan(
        prior, args_dict["sampling_size"], device
    )

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
            args_dict["save_dir"], f"{simulation_number:06}"
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
            json.dump(simulation_override_json, f, indent=4)

        # Generate a list for the command (cmd), including the Python interpreter, the script path specified with
        # 'simulator_type', and the path for the JSON override.
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
