"""
    Accuracy metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import numpy as np
import torch

from .metric_base import MetricBase


class MetricAccuracy(MetricBase):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computation of the accuracy metric.

        Args:
            output (torch.Tensor): Network output tensor (predictions).
            target (torch.Tensor): Ground truth tensor (labels).

        Returns:
            (torch.Tensor): Percentage / 100 of accurate or correct outputs (predictions that
                match the labels or ground truth).
        """

        with torch.no_grad():

            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()

        return correct / len(target)

    def __str__(self) -> str:
        """
        String representation for the accuracy metric.

        Returns:
            (str): String representation for the accuracy metric.
        """

        return "Accuracy Metric"

    def initial_value(self) -> float:
        """
        Starting value for the metric to start optimization.

        Returns:
            (float): Metric starting value set to minus infinity.
        """

        return -np.inf

    def improved(self, value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
        """
        Check if a metric value is better than other.

        Args:
            value_a (torch.Tensor): First value to compare.
            value_b (torch.Tensor): Second value to compare.

        Returns:
            (bool): True if the second value is greater than the first value, false
                otherwise.
        """

        return value_b > value_a
