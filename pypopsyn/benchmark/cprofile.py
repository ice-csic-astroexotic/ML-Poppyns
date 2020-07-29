import cProfile
import io
import os
import pathlib
import pstats


def do_cprofile(enabled, output_dir):
    def inner(func):

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
