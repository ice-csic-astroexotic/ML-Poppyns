"""
    chi square metric.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import typing

import numpy as np
import torch
import torch.nn as nn

from .metric_base import MetricBase


class MetricAccuracyCHI2(MetricBase):
    def __call__(self, output, target) -> float:

        """Computation of the accuracy metric defined as the reduced chi square value.
            The value of the reduced chi square should be near 1 for a best accuracy.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Reduced chi square value computed on a batch.
        """

        with torch.no_grad():

            red_chi2 = (
                (output - target) ** 2 / target
            ).sum() / output.data.nelement()

        return red_chi2

    def __str__(self) -> str:

        """String representation for the accuracy metric."""

        return "Reduced chi square accuracy metric"

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
