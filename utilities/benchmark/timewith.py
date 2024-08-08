"""
    TimeWith.

    This module provides a class for handling timings within a `with` scope and also
    use checkpoints inside such context.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import json
import os
import time
from typing import Optional, Self, Tuple, Type

import termcolor


class TimeWith:
    """
    Class for timing contexts or scopes with checkpointing.
    """

    def __init__(
        self,
        name: str = "",
        log_filename: str = "profile.log",
        json_filename: str = "profile.json",
        show: bool = True,
    ) -> None:
        """
        Initialization of the timing context by holding a name for it and also
        capturing the current time as the starting time for the scope.

        Args:
            name (str): A name for the context to be used when printing info.
            log_filename (str): Name of the LOG file to dump profiling information.
            json_filename (str): Name of the JSON file to dump profiling information.
            show (bool): Whether or not to print info to terminal.
        """

        self.name = name
        self.start = time.time()
        self.last = self.start
        self.log_filename = log_filename
        self.json_filename = json_filename
        self.show = show

        if self.json_filename is not None:

            if not os.path.isfile(self.json_filename):
                with open(self.json_filename, "w+") as f:
                    f.write(json.dumps({}))

            with open(self.json_filename, "r") as f:
                data = json.load(f)

            data[name] = {}
            data[name]["time"] = 0.0

            with open(self.json_filename, "w") as f:
                json.dump(data, f, indent=2)

    def elapsed(self) -> Tuple[float, float]:
        """
        Elapsed time getter since start of scope and between individual elapsed
        calls (i.e., time between checkpoints).

        Returns:
            (Tuple[float, float]): tuple that contains the cumulative time since the
                start of the context and this call and the total time spent just on
                that time window in seconds.
        """

        current = time.time()
        cumulative = current - self.start
        total = current - self.last

        # Mark the last time we fetched time with the current one for the next
        # partial timing on checkpoint.
        self.last = current

        return cumulative, total

    def checkpoint(self, name: str = "") -> None:
        """
        Checkpoints at the current time within the context optionally printing
        out the name given to the checkpoint and showing the amount of time
        elapsed since the last checkpoint (or the start of the scope if no
        checkpoint was done). Such info is also dumped to a file if a filename
        is specified.

        Args:
            name (str): A name for the checkpoint to print information.
        """

        cumulative, total = self.elapsed()
        output = "<prof>{}{} took {:.4f} [s] (cumulative {:.4f} [s])".format(
            self.name, name, total, cumulative
        ).strip()

        if self.show:

            print(termcolor.colored(output, "green"))

        if self.log_filename is not None:

            with open(self.log_filename, "a") as f:
                f.write(output + "\n")

        if self.json_filename is not None:

            with open(self.json_filename) as f:
                data = json.load(f)

            data[self.name][name] = {}
            data[self.name][name]["elapsed_time"] = total
            data[self.name][name]["cumulative_time"] = cumulative

            with open(self.json_filename, "w") as f:
                json.dump(data, f, indent=2)

    def __enter__(self) -> Self:
        """
        Enter method when a context is created.

        Returns:
            (Self): The instance of the class.
        """

        return self

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[BaseException],
    ) -> None:
        """
        Boilerplate exit method when the context is finished. In this case, it
        is overridden to optionally print the total time elapsed since its
        beginning. Such info is also dumped to a file if a filename is specified.

        Note: the signature of __exit__ is painful, forgive me for not typing
        all the arguments here.

        Args:
            type (Optional[Type[BaseException]]): The exception type if an exception occurred, else None.
            value (Optional[BaseException]): The exception instance if an exception occurred, else None.
            traceback (Optional[BaseException]): The traceback object if an exception occurred, else None.
        """

        cumulative, _ = self.elapsed()
        output = "<prof>{} {} took {:.4f} [s]".format(
            self.name, "finished", cumulative
        ).strip()

        if self.show:

            print(termcolor.colored(output, "green"))

        if self.log_filename is not None:

            with open(self.log_filename, "a") as f:
                f.write(output + "\n")

        if self.json_filename is not None:

            with open(self.json_filename, "r") as f:
                data = json.load(f)

            data[self.name]["time"] = cumulative

            with open(self.json_filename, "w") as f:
                json.dump(data, f, indent=2)
