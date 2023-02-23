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

    """ConfigurationParser"""

    def __init__(
        self, configuration, options=None, resume=None, run_id=None
    ) -> None:

        """Initialize instance."""

        # Load configuration file and apply specified options.
        self._configuration = self._update_configuration(
            configuration, options
        )

        # TODO: Not used now, will be able to resume training from checkpoint.
        self.resume = resume

        # Set save directory where the trained model will be saved.
        save_dir = pathlib.Path(self._configuration["trainer"]["save_dir"])

        # Generate a name for the experiment/run.
        run_name = self._configuration["name"]
        if run_id is None:
            run_id = datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")

        # Create directory for the save dir.
        self.save_dir = pathlib.Path().joinpath(
            save_dir, "models", run_name, run_id
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create directory for saving the log file.
        self.log_dir = pathlib.Path().joinpath(
            save_dir, "logs", run_name, run_id
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging module.
        learning_logger.setup_logging(self.log_dir)

    @classmethod
    def from_args(cls, args, options=""):

        """Initialize configuration from command line arguments."""

        # Add custom CLI options to arguments.
        for opt in options:
            args.add_argument(
                *opt.flags, default=None, type=opt.type, nargs=opt.nargs
            )

        # Parse arguments if they are not already parsed.
        if not isinstance(args, tuple):
            args = args.parse_args()

        # Load configuration from JSON file.
        configuration = learning_utils_json.read_json(
            pathlib.Path(args.configuration)
        )

        # Parse custom CLI arguments.
        modification = {
            o.target: getattr(args, _get_opt_name(o.flags)) for o in options
        }

        return cls(configuration, modification, args.weights)

    def init_object(self, name: str, module, *args, **kwargs):
        """Object handler finder.

        Finds an object handle with the provided name as type in the parsed
        configuration and gets its initialized instance with the arguments.

        Args:

            name: Name of the object to find.
            module: The Python module where the object class resides.
            args: Extra arguments for creating the instance.
            kwargs: Extra arguments for creating the instance.

        Returns:

            The object instance initialized with the provided arguments if
            the name of the requested object exists in the configuration
            dictionary. None otherwise.

        """

        if name in self._configuration:
            module_name = self._configuration[name]["type"]
            module_args = dict(self._configuration[name]["args"])
            module_args.update(kwargs)
            return getattr(module, module_name)(*args, **module_args)
        else:
            return None

    def get_logger(self, name: str, verbosity: int = 2):

        """Logger getter.

        Args:

            name: TODO: document
            verbosity: TODO: document

        Returns:

            Initialized logger with the specified name and verbosity level.

        """

        logger = logging.getLogger(name)
        logger.setLevel(learning_logger.LOG_LEVELS[verbosity])
        return logger

    def __getitem__(self, name: str):

        """Dictionary-like access to the configuration class."""
        return self._configuration[name]

    def _update_configuration(self, configuration, modifications):

        """Helper function to update configuration dictionary.

        Updates the configuration dictionary with custom CLI options. If no
        modifications are provided, the same configuration dictionary is
        returned.

        Args:

            configuration: The configuration dictionary.
            modifications: Additional parsed command line options.

        Returns:

            The updated configuration dictionary.

        """

        def _apply_update(k, v):
            if v is not None:
                keys = k.split(";")
                functools.reduce(operator.getitem, keys[:-1], configuration)[
                    keys[-1]
                ] = v

        if modifications is None:
            return configuration

        for k, v in modifications.items():
            if "," in k:
                for ki in k.split(","):
                    _apply_update(ki, v)
            else:
                _apply_update(k, v)

        return configuration


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")
