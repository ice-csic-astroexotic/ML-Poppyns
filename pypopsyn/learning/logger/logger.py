""" Logger.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import logging
import logging.config
import pathlib

import pypopsyn.learning.utils as learning_utils

LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(
    log_dir: str,
    log_config_file: str = "default_logger_config.json",
    default_level=logging.INFO,
) -> None:
    """
    Setup logging configuration

    """

    log_config_file = pathlib.Path(log_config_file)

    if log_config_file.is_file():

        log_config = learning_utils.json.read_json(log_config_file)

        # Modify logging paths based on run configuration.
        for _, handler in log_config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(log_dir / handler["filename"])

        logging.config.dictConfig(log_config)

    else:

        print("Warning: logging configuration file is not found!")
        logging.basicConfig(level=default_level)
