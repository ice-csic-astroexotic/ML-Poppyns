"""
    Logger.

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

import mlpoppyns.learning.utils.json_utils as json_utils
from mlpoppyns.simulator.config_simulator import cfg

LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(
    log_dir: str,
    log_config_file: str = "mlpoppyns/learning/logger/default_logger_config.json",
    default_level: int = logging.INFO,
) -> None:
    """
    Setup logging configuration.

    Sets up the logging subsystem by reading its configuration from a JSON
    configuration file.

    Args:
        log_dir (str): The directory to output the log files to.
        log_config_file (str): Path to the JSON configuration file.
        default_level (int): Default logging level.
    """

    path_to_software = cfg["path_to_software"]
    log_config_file = pathlib.Path().joinpath(
        path_to_software, log_config_file
    )

    if log_config_file.is_file():

        log_config = json_utils.read_json(log_config_file)

        # Modify logging paths based on run configuration.
        for _, handler in log_config["handlers"].items():
            if "filename" in handler:
                if not isinstance(log_dir, pathlib.Path):
                    log_dir = pathlib.Path(log_dir)
                handler["filename"] = str(log_dir / handler["filename"])

        logging.config.dictConfig(log_config)

    else:

        print("Warning: logging configuration file is not found!")
        print("Falling back to defaults...")
        logging.basicConfig(level=default_level)
