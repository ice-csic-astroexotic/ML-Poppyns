import time

from termcolor import colored


class timewith:
    def __init__(self, name=""):
        self.name = name
        self.start = time.time()
        self.last = self.start

    @property
    def elapsed(self):
        current = time.time()
        cumulative = current - self.start
        total = current - self.last
        self.last = current
        return cumulative, total

    def checkpoint(self, name=""):
        _, total = self.elapsed
        print(
            colored(
                "<prof>{}{} took {:.4f} [s]".format(
                    self.name, name, total
                ).strip(),
                "green",
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        cumulative, _ = self.elapsed
        print(
            colored(
                "<prof>{} {} took {:.4f} [s]".format(
                    self.name, "finished", cumulative
                ).strip(),
                "green",
            )
        )
        pass
