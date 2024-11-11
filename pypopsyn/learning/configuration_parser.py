"""
    Configuration parser.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import datetime
import functools
import logging
import operator
import pathlib
from collections import OrderedDict
from logging import Logger
from typing import Any, Optional, Union

import pypopsyn.learning.logger.logger as learning_logger
import pypopsyn.learning.utils.json as learning_utils_json


class ConfigurationParser:
    """
    ConfigurationParser
    """

    def __init__(
        self,
        configuration: OrderedDict,
        infer: bool,
        options: Optional[dict] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize instance.

        Args:
            configuration (OrderedDict): The configuration dictionary.
            infer (bool): Boolean indicating if inference mode is on.
            options (Optional[dict]): Optional dictionary with additional options. Default None.
            run_id (Optional[str]): Optional string identifier for the run. Default None.
        """

        # Load configuration file and apply specified options.
        self._configuration = self._update_configuration(
            configuration, options
        )
        # If resuming training from a previous run, do not create new save_dir and log_dir folders.
        # Instead, use the same directories from the previous run.
        self.resume = self._configuration["resume_training"]["resume"]

        if self.resume:
            self.save_dir = self._configuration["resume_training"]["save_dir"]
            self.log_dir = self._configuration["resume_training"]["log_dir"]
        else:
            # Generate a name for the experiment/run.
            run_name = self._configuration["name"]
            if run_id is None:
                run_id = datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")

            # Set save directory where the trained model or the inference results will be saved.
            if infer:
                save_dir = pathlib.Path(
                    self._configuration["infer"]["save_dir"]
                )
            else:
                save_dir = pathlib.Path(
                    self._configuration["trainer"]["save_dir"]
                )

                # Create directory for saving the model.
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
    def from_args(
        cls, args: argparse.Namespace, options: list = ""
    ) -> "ConfigurationParser":
        """
        Initialize configuration from command line arguments.

        Args:
            args (argparse.Namespace): Command-line arguments parsed by argparse.
            options (list): Options parsed by argparse. Default is an empty list.

        Returns:
            (ConfigurationParser): An instance of the ConfigurationParser class.
        """

        if isinstance(args, argparse.ArgumentParser):
            # Add custom CLI options to arguments.
            for opt in options:
                args.add_argument(
                    *opt.flags, default=None, type=opt.type, nargs=opt.nargs
                )
            parsed_args = args.parse_args()
        else:
            # args is already a Namespace.
            parsed_args = args

        # Load configuration from JSON file.
        configuration = learning_utils_json.read_json(
            parsed_args.configuration
        )

        # Parse custom CLI arguments.
        modification = {
            o.target: getattr(parsed_args, _get_opt_name(o.flags))
            for o in options
        }

        return cls(configuration, parsed_args.infer, modification)

    def init_object(
        self, name: str, module: Any, *args: Any, **kwargs: Any
    ) -> Union[Any, None]:
        """
        Object handler finder.

        Finds an object handle with the provided name as type in the parsed
        configuration and gets its initialized instance with the arguments.

        Args:
            name (str): Name of the object to find.
            module (Any): The Python module where the object class resides.
            args (Any): Extra arguments for creating the instance.
            kwargs (Any): Extra arguments for creating the instance.

        Returns:
            (Union[Any, None]): The object instance initialized with the provided arguments if the name of the requested
                object exists in the configuration dictionary. None otherwise.
        """

        if name in self._configuration:
            module_name = self._configuration[name]["type"]
            module_args = dict(self._configuration[name]["args"])
            module_args.update(kwargs)
            return getattr(module, module_name)(*args, **module_args)
        else:
            return None

    def get_logger(self, name: str, verbosity: int = 2) -> Logger:
        """
        Logger getter.

        Args:
            name (str): Name for the logger.
            verbosity (int): Logging level. By default it is set to INFO.

        Returns:
            (Logger): Initialized logger with the specified name and verbosity level.
        """

        logger = logging.getLogger(name)
        logger.setLevel(learning_logger.LOG_LEVELS[verbosity])
        return logger

    def __getitem__(self, name: str) -> Any:
        """
        Dictionary-like access to the configuration class.

        Args:
            name (str): Name of the configuration key.

        Returns:
            (Any): The value associated with the given key in the configuration.
        """
        return self._configuration[name]

    def _update_configuration(
        self, configuration: OrderedDict, modifications: dict
    ) -> OrderedDict:

        """
        Helper function to update configuration dictionary.

        Updates the configuration dictionary with custom CLI options. If no
        modifications are provided, the same configuration dictionary is
        returned.

        Args:
            configuration (OrderedDict): The configuration dictionary.
            modifications (dict): Additional parsed command line options.

        Returns:
            (OrderedDict): The updated configuration dictionary.
        """

        def _apply_update(k: str, v: Any) -> None:
            """
            Updates a nested dictionary using a semicolon separated key string.

            Args:
                k (str): A semicolon separated string representing the keys to traverse in the dictionary.
                v (Any): The value to set at the specified location in the dictionary.
            """
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


def _get_opt_name(flags: list) -> str:
    """
    Extract the option name from the flags.

    Args:
        flags (list): List of command line flags.

    Returns:
        (str): The option name extracted from the flags.
    """
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def str_to_bool(value: str | None) -> bool:
    """
    Convert a string value to a boolean.

    This function is intended for parsing command-line boolean arguments that accept "True" or "False" as strings.

    Args:
        value (str | None): The string to be converted to a boolean.
            Accepts "True", "False", "1", "0" (case-insensitive) or None.

    Returns:
        (bool): The corresponding boolean value. If "True" or "1" is passed,
              returns True. If "False" or "0" is passed, returns False.
    """

    if isinstance(value, str):
        # Handle "True" or "1".
        if value.lower() in ("true", "1"):
            return True
        # Handle "False" or "0".
        elif value.lower() in ("false", "0"):
            return False

    # Raise an error if the input is invalid.
    raise argparse.ArgumentTypeError(f"Boolean value expected. Got '{value}'")
