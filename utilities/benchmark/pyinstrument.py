"""
PyInstrument helper module.

This module provides helper functions to perform deep profiling of other routines
using Python's third-party PyInstrument.

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

import io
import os
import pathlib
import pstats
import typing

import pyinstrument


def profile(enabled: bool = True, show: bool = True, output_dir: str = None):
    """
    Function to be used as decorator to perform a deep PyInstrument of another
    routine. It will call such function with the provided arguments with
    profiling enabled, later it gathers all the results in a readable format
    sorting them by total time, and then outputs the PyInstrument result to a
    text file in the specified folder with the name of the function as file name.

    Args:

        enabled (bool): Whether or not profiling is toggled.

        show (bool): Whether or not to print info to terminal.

        output_dir (str): Output directory for the profile text file.

    Returns:

        If profiling is disabled, it just returns the result of the function
        without profiling and generating any text file (seamless execution). If
        profiling is enabled, it also returns the result of executing the
        function seamlessly but generates the text output as specified above.
    """

    def inner(func: typing.Callable):

        if not enabled:
            return func

        def profiled_func(*args, **kwargs):

            profiler = pyinstrument.Profiler()

            try:

                profiler.start()
                result = func(*args, **kwargs)
                profiler.stop()
                return result

            finally:

                output = profiler.output_text(unicode=True, color=True)

                if show:

                    print(output)

                if output_dir is not None:

                    path = pathlib.Path(output_dir)
                    path.mkdir(exist_ok=True)
                    filename = path / (func.__name__ + ".pyinstrument")

                    with open(filename, "w") as f:
                        f.write(output)

        return profiled_func

    return inner
