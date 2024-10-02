"""
    chi square metric.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import numpy as np
import torch

from .metric_base import MetricBase


class MetricAccuracyCHI2(MetricBase):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computation of the accuracy metric defined as the reduced chi square value.
        The value of the reduced chi square should be near 1 for best accuracy.

        Args:
            output (torch.Tensor): Network output tensor (predictions).
            target (torch.Tensor): Ground truth tensor (labels).

        Returns:
            (torch.Tensor): Reduced chi square value computed on a batch.
        """

        with torch.no_grad():

            red_chi2 = (
                (output - target) ** 2 / target
            ).sum() / output.data.nelement()

        return red_chi2

    def __str__(self) -> str:
        """
        String representation for the chi squared metric.

        Returns:
            (str): String representation for the chi squared metric.
        """

        return "Reduced chi square accuracy metric"

    def initial_value(self) -> float:
        """
        Starting value for the metric to start optimization.

        Returns:
            (float): Starting value for the metric to start optimization.
        """

        return np.inf

    def improved(self, value_a: torch.Tensor, value_b: torch.Tensor) -> bool:
        """
        Check if a metric value is better than the other.

        Args:
            value_a (torch.Tensor): First value to compare (current value).
            value_b (torch.Tensor): Second value to compare (new value).

        Returns:
            (bool): True if the second value is lower than the first value, false
                otherwise.
        """

        return value_b < value_a
