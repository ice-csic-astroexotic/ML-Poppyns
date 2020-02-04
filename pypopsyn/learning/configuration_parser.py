""" Configuration Parser.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import datetime
import functools
import json
import logging
import operator
import os
import pathlib

import pypopsyn.learning.logger.logger as learning_logger
import pypopsyn.learning.utils.json as learning_utils_json


class ConfigurationParser:

    """ ConfigurationParser """

    def __init__(self, configuration, options=None, resume=None) -> None:

        """ Initialize instance. """

        # Load configuration file and apply specified options.
        self._configuration = self._update_configuration(
            configuration, options
        )

        # Generate a name for the experiment/run.
        run_name = self._configuration["name"]
        run_id = datetime.datetime.now().strftime(r"%m%d_%H%M%S")

        # Create directory for saving the log file.
        # self._log_dir = os.path.join("logs", run_name, run_id)
        self._log_dir = pathlib.Path().joinpath("logs", run_name, run_id)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging module.
        learning_logger.setup_logging(self._log_dir)

    @classmethod
    def from_args(cls, args):

        """ Initialize configuration from command line arguments. """

        # Parse arguments.
        if not isinstance(args, tuple):
            args = args.parse_args()

        # Load configuration from JSON file.
        configuration = learning_utils_json.read_json(args.configuration)

        return cls(configuration)

    def init_object(self, name, module, *args, **kwargs):

        """ Object handler finder.

            Finds an object handle with the provided name as type in the parsed
            configuration and gets its initialized instance with the arguments.

            Args:

                name Name of the object to find.
                module: TODO: document.
                args: TODO: document.
                kwargs: TODO: document.

            Returns:

                The object instance intialized with the provided arguments.

        """

        module_name = self._configuration[name]["type"]
        module_args = dict(self._configuration[name]["args"])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def get_logger(self, name: str, verbosity: int = 2):

        """ Logger getter.

            Args:

                name: TODO: document
                verbosity: TODO: document

            Returns:

                Initialized logger with the specified name and verbosity level.

        """

        logger = logging.getLogger(name)
        logger.setLevel(learning_logger.LOG_LEVELS[verbosity])
        return logger

    def _update_configuration(self, configuration, modifications):

        """ Helper function to update configuration dictionary.

            Updates the configuration dictionary with custom CLI options.

            Args:

                configuration: The configuration dictionary.
                modifications: Additional command line options.

            Returns:

                The updated configuration dictionary.

        """

        if modifications is None:
            return configuration

        for k, v in modifications.items():
            if v is not None:
                keys = k.split(";")
                functools.reduce(operator.getitem, keys[:-1], configuration)[
                    keys[-1]
                ] = v
