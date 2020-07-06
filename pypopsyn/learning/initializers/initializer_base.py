import abc


class InitializerBase:

    """ Base abstract class for all weight initializers. """

    @abc.abstractmethod
    def __call__(self, m):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError
