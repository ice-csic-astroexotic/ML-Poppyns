""" Base metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import abc


class MetricBase:

    """ Base abstract class for all metrics. """

    @abc.abstractmethod
    def __call__(self, output, target):

        """ Actual computation of the metric function. """

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):

        """ String representation of the metric. """

        raise NotImplementedError
