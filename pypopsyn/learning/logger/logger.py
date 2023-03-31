""" Logger.

    Utility functions for setting up and dealing with the logging subsystem in
    order to generate messages both to the console and to log useful info of
    the process to output files.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import logging
import logging.config
import pathlib

import pypopsyn.learning.utils as learning_utils
from pypopsyn.simulator.configuration import cfg

LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(
    log_dir: str,
    log_config_file: str = "examples/learning/default_logger_config.json",
    default_level=logging.INFO,
) -> None:
    """
    Setup logging configuration.

    Sets up the logging subsystem by reading its configuration from a JSON
    configuration file.

    Args:
        log_dir: The directory to output the log files to.
        log_config_file: Path to the JSON configuration file.
        default_level: Default logging level.

    Returns:
        Nothing.

    """

    path_server_software = cfg["path_server_software"]
    log_config_file = pathlib.Path().joinpath(
        path_server_software, log_config_file
    )

    if log_config_file.is_file():

        log_config = learning_utils.json.read_json(log_config_file)

        # Modify logging paths based on run configuration.
        for _, handler in log_config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(log_dir / handler["filename"])

        logging.config.dictConfig(log_config)

    else:

        print("Warning: logging configuration file is not found!")
        print("Falling back to defaults...")
        logging.basicConfig(level=default_level)
