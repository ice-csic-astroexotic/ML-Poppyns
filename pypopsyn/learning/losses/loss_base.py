"""
    Base loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc


class LossBase:

    """Base abstract class for all losses."""

    @abc.abstractmethod
    def __call__(self, output, target):

        """Actual computation of the loss function."""

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):

        """String representation of the model."""

        raise NotImplementedError
