"""
    Base metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc
import enum

import numpy as np
import torch


class MetricBehavior(enum.Enum):
    MIN = 0
    MAX = 1

    initial_values = {MIN: np.inf, MAX: -np.inf}


class MetricBase:
    """
    Base abstract class for all evaluation metrics.
    """

    @abc.abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the metric value based on the model output and target.

        This method takes the predicted output from the model and the
        corresponding target values, and computes the metric score. The specific
        implementation of this method will vary depending on the type of metric
        being implemented (e.g., Accuracy, Mean relative error...).

        Args:
            output (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth values to compare against.

        Returns:
            (float): The computed metric value.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        String representation of the metric.

        This method should provide a human-readable description of the metric,
        including its name and any relevant parameters or characteristics.

        Returns:
            (str): A string that describes the metric.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def initial_value(self) -> float:
        """
        Return the starting value for the metric optimization.

        This method should return the initial value that the metric should start
        with for optimization purposes. For example, in the case of accuracy,
        the initial value could be 0.0, as we want to maximize the accuracy.

        Returns:
            (float): The initial value for the metric optimization.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def improved(self, value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
        """
        Check if the metric value has improved.

        This method compares two metric values and determines whether the second
        value represents an improvement over the first value. The specific
        implementation of this method will depend on the type of metric being
        used (e.g., for accuracy, a higher value is better, while for loss, a
        lower value is better).

        Args:
            value_a (float): The first metric value to compare.
            value_b (float): The second metric value to compare.

        Returns:
            (bool): True if the second value represents an improvement over the
                first value, False otherwise.
        """

        raise NotImplementedError
