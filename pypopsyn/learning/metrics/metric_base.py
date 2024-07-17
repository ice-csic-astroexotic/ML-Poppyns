"""
    Base metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc
import enum
import typing

import numpy as np
import torch


class MetricBehavior(enum.Enum):
    MIN = 0
    MAX = 1

    initial_values = {MIN: np.inf, MAX: -np.inf}


class MetricBase:
    """
    Base abstract class for all metrics.
    """

    @abc.abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:

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
        """
        Starting value for the metric to start optimization.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def improved(self, value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
        """
        Check if the metric value has improved.
        """

        raise NotImplementedError
