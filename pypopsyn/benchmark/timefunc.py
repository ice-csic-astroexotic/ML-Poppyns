import time

from termcolor import colored


def timefunc(f):
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
