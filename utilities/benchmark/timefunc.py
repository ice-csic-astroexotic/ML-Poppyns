"""
    TimeFunc.

    This module provides a function for decorating other subroutines to automatically
    obtain timings for them.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import time
import typing

import termcolor


def time_function(filename: str = None, show: bool = True) -> typing.Callable:
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
        (typing.Callable): The result of calling the specified function.
    """

    def inner(func: typing.Callable) -> typing.Callable:
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
