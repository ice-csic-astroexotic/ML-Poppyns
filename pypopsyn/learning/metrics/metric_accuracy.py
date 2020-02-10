""" Accuracy metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import numpy as np
import torch

from .metric_base import MetricBase


class MetricAccuracy(MetricBase):
    def __call__(self, output, target) -> float:

        """ Computation of the accuracy metric.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Percentage / 100 of accurate or correct outputs (predictions that
            match the labels or ground truth).

        """

        with torch.no_grad():

            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()

        return correct / len(target)

    def __str__(self) -> str:

        """ String representation for the Accuracy metric. """

        return "Accuracy Metric"

    def initial_value(self) -> float:

        """ Starting value for the metric to start optimization. """

        return -np.inf

    def improved(self, value_a, value_b) -> bool:

        """ Check if a metric value is better than other.

        Args:
            value_a: First value to compare.
            value_b: Second value to compare.

        Returns:
            True if the second value is greater than the first value, false
            otherwise.

         """

        return value_b > value_a
