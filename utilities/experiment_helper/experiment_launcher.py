"""
    Experiment launcher script.

    This script helps running multiple experiments by taking a list of commands and
    executing them with a process pool. This way, a huge list of experiments can
    be left running unattended.

    Display help message to run the code:

    python experiment_launcher.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import argparse
import logging
import multiprocessing as mp
import pathlib
import subprocess
import sys
import threading
import typing

log = logging.getLogger(__name__)


def run_experiment(command: str) -> typing.Tuple[pathlib.Path, str]:
    """
    Run experiment command.

    This is the main routine for running a particular experiment. It runs the
    provided experiment command (a Python call to the experiment script with a
    set of CLI arguments) and captures all the output of the process.

    Args:
        command (str): full command to execute the experiment.

    Returns:
        (Tuple[pathlib.Path, str]): The experiment command and the convoluted output of the process.
    """

    # Acquire the lock and block any other process from executing
    # for two seconds. We do this in order not to launch two processes
    # at the exact same second so that their output folders (which are
    # named automatically MMDD_HHMMSS) are not overwritten.
    starting.acquire()
    threading.Timer(2, starting.release).start()

    # Once the process has released the lock for another process to wait
    # it can proceed with the execution of the experiment.
    log.info("Launching experiment {}".format(command))

    process_output = subprocess.check_output(
        command, stderr=subprocess.STDOUT, shell=True
    )

    log.info("Experiment finished...")

    return command, process_output.decode("utf-8")


def log_experiment(process_result: typing.Tuple[pathlib.Path, str]) -> None:
    """
    Callback to log all the info returned from an experiment run.

    Args:
        process_result (Tuple[pathlib.Path, str]): tuple containing the process experiment command and the
            whole process output to console string.
    """

    log.info("")
    log.info(
        "****************************************************************"
    )
    log.info('Ran experiment "{}"!'.format(process_result[0]))
    log.info("Process output:\n {}".format(process_result[1]))
    log.info("Process finished...")


def setup_process_pool(event: mp.Event, lock: mp.Lock) -> None:
    """
    Set up the process pool for multiprocessing with a global pause/resume event.

    Args:
        event (mp.Event): reference to a master process event that will signal the child
            processes to pause or resume execution.
        lock (mp.lock): a reference to a master process lock that will coordinate the
            child process launching with waiting times.
    """

    global unpaused
    unpaused = event

    global starting
    starting = lock


def main(args) -> None:

    # Event on the master process that will be used to synchronize the child
    # processes and signal them for execution in the pool.
    event = mp.Event()
    # Lock on the master process to impose a delay in the process execution
    # so that none of them can be launched exactly at the same time.
    lock = mp.Lock()
    # A pool of processes with a defined capacity, a process spawning setup
    # routine and a general event to signal process execution.
    pool = mp.Pool(
        args.processes,
        setup_process_pool,
        (
            event,
            lock,
        ),
    )

    # Read the command list file, each command should be one single line.
    with open(args.command_list) as f:
        commands = f.readlines()
    commands = [x.strip() for x in commands]

    # Fill the pool with one process for each command in the list. Each one
    # of them will execute the experiment subroutine with the specified
    # command and will log its results upon completion.
    for command in commands:

        pool.apply_async(
            run_experiment, args=(command,), callback=log_experiment
        )

        log.info("Experiment process sent to pool for execution...")

    log.info("")
    log.info("***************************************************************")
    log.info("Launching experiments")
    log.info("***************************************************************")

    # Signal the processes to begin execution in the pool.
    event.set()

    # Wait for all processes to finish.
    pool.close()
    pool.join()


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyPopSyn Experiment Launcher")

    args.add_argument(
        "--command_list",
        nargs="?",
        type=str,
        default="utilities/experiment_helper/command_list.txt",
        help="List of commands to execute.",
    )
    args.add_argument(
        "--processes",
        nargs="?",
        type=int,
        default=1,
        help="Number of simultaneous processes for the pool.",
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
