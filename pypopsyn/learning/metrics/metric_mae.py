"""
    Mean absolute error accuracy metric.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
"""

import typing

import numpy as np
import torch
import torch.nn as nn

from .metric_base import MetricBase


class MetricAccuracyMAE(MetricBase):
    def __call__(self, output, target) -> float:

        """Computation of the accuracy metric defined as mean absolute error.
            The value of the MAE should be 0 for the best accuracy.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Mean Absolute Error computed over a batch.
        """

        with torch.no_grad():

            self.mae = nn.L1Loss()
            mae = self.mae(output, target)

        return mae

    def __str__(self) -> str:

        """String representation for the accuracy metric."""

        return "Mean Absolute Error accuracy metric"

    def initial_value(self) -> float:

        """Starting value for the metric to start optimization."""

        return np.inf

    def improved(self, value_a, value_b) -> bool:

        """Check if a metric value is better than other.

        Args:
            value_a: First value to compare (current value).
            value_b: Second value to compare (new value).

        Returns:
            True if the second value is lower than the first value, false
            otherwise.

        """

        return value_b < value_a
