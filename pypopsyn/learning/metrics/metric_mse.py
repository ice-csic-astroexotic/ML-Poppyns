"""
    Mean square error metric.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
"""

import numpy as np
import torch
import torch.nn as nn

from .metric_base import MetricBase


class MetricAccuracyMSE(MetricBase):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:

        """
        Computation of the metric defined as mean square error.
        The value of the MSE should be 0 for the best accuracy.

        Args:
            output (torch.Tensor): Network output tensor (predictions).
            target (torch.Tensor): Ground truth tensor (labels).

        Returns:
            (torch.Tensor): Mean square error computed over a batch.
        """

        with torch.no_grad():

            self.mse = nn.MSELoss()
            mse = self.mse(output, target)

        return mse

    def __str__(self) -> str:

        """
        String representation for the accuracy metric.

        Returns:
            (str): String representation for the accuracy metric.
        """

        return "Mean square error accuracy metric"

    def initial_value(self) -> float:
        """
        Starting value for the metric to start optimization.

        Returns:
            (float): Starting value for the metric to start optimization.
        """

        return np.inf

    def improved(self, value_a: torch.Tensor, value_b: torch.Tensor) -> bool:

        """
        Check if a metric value is better than other.

        Args:
            value_a (torch.Tensor): First value to compare (current value).
            value_b (torch.Tensor): Second value to compare (new value).

        Returns:
            (bool): True if the second value is lower than the first value, false
                otherwise.
        """

        return value_b < value_a
