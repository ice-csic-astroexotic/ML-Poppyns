"""
TimeFunc.

This module provides a function for decorating other subroutines to automatically
obtain timings for them.

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
import typing

import termcolor


def time_function(filename: str = None, show: bool = True):
    """
    Function to use as a decorator to time another function for each call
    seamlessly. It sets up a timer, calls the specified function with the
    provided arguments and prints the elapsed time, returning the result
    of the provided function call.

    The profiling result is optionally printed to screen and dumped to a file
    if a filename is specified.

    Args:

        filename (str): Name of the file to dump profiling information.

        show (bool): Whether or not to print info to terminal.

    Returns:

        The result of calling the specified function.
    """

    def inner(func: typing.Callable):
        def f_timer(*args, **kwargs):

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            output = "<prof>{} took {:.4f} [s]".format(
                func.__name__, end - start
            )

            if show:

                print(termcolor.colored(output, "green"))

            if filename is not None:

                with open(filename, "a") as f:
                    f.write(output + "\n")

            return result

        return f_timer

    return inner
