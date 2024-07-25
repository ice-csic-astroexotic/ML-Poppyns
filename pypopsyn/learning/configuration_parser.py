"""
    Configuration parser.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import argparse
import datetime
import functools
import json
import logging
import operator
import os
import pathlib
from typing import Any, Dict, List, Optional

import pypopsyn.learning.logger.logger as learning_logger
import pypopsyn.learning.utils.json as learning_utils_json


class ConfigurationParser:
    """
    ConfigurationParser
    """

    def __init__(
        self,
        configuration: dict,
        infer: bool,
        options: Optional[Dict] = None,
        resume: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize instance.

        Args:
            configuration (dict): The configuration dictionary.
            infer (bool): Whether to run in inference mode.
            options (Optional[Dict]): Modifications to apply to the configuration.
            resume (Optional[str]): Path to the checkpoint to resume training.
            run_id (Optional[str]): Unique identifier for the run.
        """

        # Load configuration file and apply specified options.
        self._configuration = self._update_configuration(
            configuration, options
        )

        # TODO: Not used now, will be able to resume training from checkpoint.
        self.resume = resume

        # Generate a name for the experiment/run.
        run_name = self._configuration["name"]
        if run_id is None:
            run_id = datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")

        # Set save directory where the trained model or the inference results will be saved.
        if infer:
            save_dir = pathlib.Path(self._configuration["infer"]["save_dir"])
        else:
            save_dir = pathlib.Path(self._configuration["trainer"]["save_dir"])

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
        cls, args: Any, options: Optional[List] = ""
    ) -> "ConfigurationParser":
        """
        Initialize configuration from command line arguments.

        Args:
            args (Any): Parsed command line arguments.
            options (Optional[List[Any]]): Custom CLI options to add to arguments.

        Returns:
            (ConfigurationParser): An instance of ConfigurationParser.
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

        return cls(
            configuration,
            parsed_args.infer,
            modification,
            parsed_args.trained_model,
        )

    def init_object(
        self, name: str, module: Any, *args: Any, **kwargs: Any
    ) -> Optional[Any]:
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
            (Optional[Any]): The object instance initialized with the provided arguments if
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

    def get_logger(self, name: str, verbosity: int = 2) -> logging.Logger:
        """
        Logger getter.

        Args:
            name (str): Name for the logger.
            verbosity (int): Logging level. By default it is set to INFO.

        Returns:
            logging.Logger: Initialized logger with the specified name and verbosity level.
        """

        logger = logging.getLogger(name)
        logger.setLevel(learning_logger.LOG_LEVELS[verbosity])
        return logger

    def __getitem__(self, name: str) -> Any:
        """
        Dictionary-like access to the configuration class.

        Args:
            name (str): The key of the configuration item.

        Returns:
            (Any): The value corresponding to the given key.
        """
        return self._configuration[name]

    def _update_configuration(
        self, configuration: dict, modifications: dict
    ) -> dict:
        """
        Helper function to update configuration dictionary.

        Updates the configuration dictionary with custom CLI options. If no
        modifications are provided, the same configuration dictionary is
        returned.

        Args:
            configuration (dict): The configuration dictionary.
            modifications (dict): Additional parsed command line options.

        Returns:
            (dict): The updated configuration dictionary.
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


def _get_opt_name(flags: List[str]) -> str:
    """
    Extracts the option name from a list of command line flags.

    This function looks for a flag that starts with '--' and returns it
    without the leading '--'. If no such flag is found, it returns the
    first flag in the list without the leading '--'.

    Args:
        flags (List[str]): A list of command line flags.

    Returns:
        (str): The extracted option name without the leading '--'.
    """
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")
