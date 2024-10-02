"""
    cProfile helper module.

    This module provides helper functions to perform deep profiling of other routines
    using Python's built-in cProfiler.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import cProfile
import io
import pathlib
import pstats
import typing


def do_cprofile(enabled: bool, output_dir: str) -> typing.Callable:
    """
    Function to be used as decorator to perform a deep cProfile of another
    routine. It will call such function with the provided arguments with
    profiling enabled, later it gathers all the results in a readable format
    sorting them by total time, and then outputs the cProfile result to a text
    file in the specified folder with the name of the function as file name.

    Args:
        enabled (bool): Whether or not profiling is toggled.
        output_dir (str): Output directory for the profile text file.

    Returns:
        (typing.Callable): If profiling is disabled, it just returns the result of the function
            without profiling and generating any text file (seamless execution). If
            profiling is enabled, it also returns the result of executing the
            function seamlessly but generates the text output as specified above.
    """

    def inner(func: typing.Callable) -> typing.Callable:

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
