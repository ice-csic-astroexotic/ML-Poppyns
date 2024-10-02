"""
    PyInstrument helper module.

    This module provides helper functions to perform deep profiling of other routines
    using Python's third-party PyInstrument.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import pathlib
import typing

import pyinstrument


def profile(
    enabled: bool = True, show: bool = True, output_dir: str = None
) -> typing.Callable:
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
        (typing.Callable): If profiling is disabled, it just returns the result of the function
            without profiling and generating any text file (seamless execution). If
            profiling is enabled, it also returns the result of executing the
            function seamlessly but generates the text output as specified above.
    """

    def inner(func: typing.Callable) -> typing.Callable:

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
