"""
    Base metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc
import enum
import typing

import numpy as np


class MetricBehavior(enum.Enum):
    MIN = 0
    MAX = 1

    initial_values = {MIN: np.inf, MAX: -np.inf}


class MetricBase:

    """Base abstract class for all metrics."""

    @abc.abstractmethod
    def __call__(self, output, target) -> float:

        """
        Actual computation of the metric function.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:

        """
        String representation of the metric.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def initial_value(self) -> float:

        """Starting value for the metric to start optimization."""

        raise NotImplementedError

    @abc.abstractmethod
    def improved(self, value_a, value_b) -> bool:

        """
        Check if the metric value has improved.

        Args:
            value_a (torch.Tensor): First value to compare.
            value_b (torch.Tensor): Second value to compare.

        Returns:
            (bool): True if the second value is better than the first, false otherwise.

        """

        raise NotImplementedError
