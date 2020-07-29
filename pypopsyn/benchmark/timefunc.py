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

from termcolor import colored


def time_function(f: typing.Callable):
    """
    Function to use as a decorator to time another function for each call
    seamlessly. It sets up a timer, calls the specified function with the
    provided arguments and prints the elapsed time, returning the result
    of the provided function call.

    Args:

        f (typing.Callable): function to be called and timed.

    Returns:

        The result of calling the specified function.
    """

    def f_timer(*args, **kwargs):

        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()

        print(
            colored(
                "<prof>{} took {:.4f} [s]".format(f.__name__, end - start),
                "green",
            )
        )

        return result

    return f_timer
