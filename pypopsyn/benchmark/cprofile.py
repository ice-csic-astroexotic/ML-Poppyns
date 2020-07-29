"""
cProfile helper module.

This module provides helper functions to perform deep profiling of other routines
using Python's built-in cProfiler.

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

import cProfile
import io
import os
import pathlib
import pstats
import typing


def do_cprofile(enabled: bool, output_dir: str):
    """
    Function to be used as decorator to perform a deep cProfile of another
    routine. It will call such function with the provided arguments with
    profiling enabled, later it gathers all the results in a readable format
    sorting them by total tiem, and then outputs the cProfile result to a text
    file in the specified folder with the name of the function as file name.

    Args:

        enabled (bool): Whether or not profiling is toggled.

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

            profile = cProfile.Profile()

            try:

                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result

            finally:

                s = io.StringIO()
                ps = pstats.Stats(profile, stream=s).sort_stats("tottime")
                ps.print_stats()

                path = pathlib.Path(output_dir)
                path.mkdir(exist_ok=True)
                filename = path / (func.__name__ + ".txt")

                with open(filename, "w") as f:
                    f.write(s.getvalue())

        return profiled_func

    return inner
