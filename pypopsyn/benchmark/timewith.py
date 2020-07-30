"""
TimeWith.

This module provides a class for handling timings within a `with` scope and also
use checkpoints inside such context.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time

import termcolor


class TimeWith:
    """
    Class for timing contexts or scopes with checkpointing.
    """

    def __init__(
        self, name: str = "", filename: str = None, show: bool = True
    ) -> None:
        """
        Initialization of the timing context by holding a name for it and also
        capturing the current time as the starting time for the scope.

        Args:

            name (str): A name for the context to be used when printing info.

            filename (str): Name of the file to dump profiling information.

            show (bool): Whether or not to print info to terminal.

        Returns:

            Nothing.

        """

        self.name = name
        self.start = time.time()
        self.last = self.start
        self.filename = filename
        self.show = show

    def elapsed(self) -> None:
        """
        Elapsed time getter since start of scope and between individual elapsed
        calls (i.e., time between checkpoints).

        Returns:

            A tuple (float, float) that contains the cumulative time since the
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

        Returns:

            Nothing.

        """

        cumulative, total = self.elapsed()
        output = "<prof>{}{} took {:.4f} [s] (cumulative {:.4f} [s])".format(
            self.name, name, total, cumulative
        ).strip()

        if self.show:

            print(termcolor.colored(output, "green"))

        if self.filename is not None:

            with open(self.filename, "a") as f:
                f.write(output + "\n")

    def __enter__(self):
        """
        Enter method when a context is created.

        Returns:

            Self.
        """

        return self

    def __exit__(self, type, value, traceback):
        """
        Boilerplate exit method when the context is finished. In this case, it
        is overridden to optionally print the total time elapsed since its
        beginning. Such info is also dumped to a file if a filename is specified.

        Note: the signature of __exit__ is painful, forgive me for not typing
        all the arguments here.

        Returns:

            Nothing.
        """

        cumulative, _ = self.elapsed()
        output = "<prof>{} {} took {:.4f} [s]".format(
            self.name, "finished", cumulative
        ).strip()

        if self.show:

            print(termcolor.colored(output, "green"))

        if self.filename is not None:

            with open(self.filename, "a") as f:
                f.write(output + "\n")
