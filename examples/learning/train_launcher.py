
#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Training launcher script.

    This script helps running training by taking a list of training commands and
    executing them with a process pool. This way, a huge list of experiments can
    be left running unattended.

    Running the code:

        python3 examples/learning/train_launcher.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import logging
import multiprocessing as mp
import pathlib
import subprocess
import sys
import typing

log = logging.getLogger(__name__)


def run_experiment(
      command: str
) -> typing.Tuple[pathlib.Path, str]:
    """
    Run experiment comamnd.

    This is the main routine for running a particular experiment. It runs the
    provided experiment command (a Python call to the training script with a
    set of CLI arguments) and captures all the output of the process.

    Args:
        command: full command to execute the experiment.

    Returns:
        The experiment command and the consolute output of the process.

    """

    log.info("Launching experiment {}".format(command))

    process_output = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        shell=True
    )

    log.info("Experiment finished...")

    return command, process_output.decode("utf-8")


def log_experiment(
    process_result: typing.Tuple[pathlib.Path, str]
) -> None:

    """
    Callback to log all the info returned from an experiment run.

    Args:
        process_result: tuple containing the process experiment command and the
          whole process output to console string.

    Returns:
        Nothing.

    """

    log.info("")
    log.info("****************************************************************")
    log.info("Ran experiment \"{}\"!".format(process_result[0]))
    log.info("Process output:\n {}".format(process_result[1]))
    log.info("Process finished...")


def setup_process_pool(
    event: mp.Event
) -> None:

    """
    Setup the process pool for multiprocessing with a global pause/resume event.
    """

    global unpaused
    unpaused = event


def main(args):

    event = mp.Event()
    pool = mp.Pool(args.processes, setup_process_pool, (event, ))

    with open(args.command_list) as f:
        commands = f.readlines()
    commands = [x.strip() for x in commands] 

    for command in commands:

        pool.apply_async(
          run_experiment,
          args=(command,),
          callback=log_experiment
        )

        log.info("Experiment process sent to pool for execution...")

    log.info("")
    log.info("***************************************************************")
    log.info("Launching experiments")
    log.info("***************************************************************")

    event.set()
    pool.close()
    pool.join()


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="PyPopSyn Training Launcher"
    )

    args.add_argument(
        "--command_list",
        nargs="?",
        type=str,
        default="examples/learning/command_list.txt",
        help="List of commands to execute for training"
    )
    args.add_argument(
        "--processes",
        nargs="?",
        type=int,
        default=1,
        help="Number of simultaneous processes"
    )

    args = args.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    main(args)
