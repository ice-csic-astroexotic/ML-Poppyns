""" RMSE Accuracy metric.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)


    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import numpy as np
import torch
import torch.nn as nn

from .metric_base import MetricBase


class MetricAccuracyRMSE(MetricBase):
    def __call__(self, output, target, eps=1e-6) -> float:

        """ Computation of the accuracy metric defined as root mean squared error.
            The value of the RMSE should be 0 for the best accuracy.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Root Mean Squared Error computed over a batch.
        """

        self.eps = eps
        rmse = 0.0

        with torch.no_grad():

            self.mse = nn.MSELoss()
            # adding a small epsilon to avoid null values
            rmse = torch.sqrt(self.mse(output, target) + self.eps)

        return rmse

    def __str__(self) -> str:

        """ String representation for the Accuracy metric. """

        return "Root Mean Squared Error accuracy metric"

    def initial_value(self) -> float:

        """ Starting value for the metric to start optimization. """

        return np.inf

    def improved(self, value_a, value_b) -> bool:

        """ Check if a metric value is better than other.

        Args:
            value_a: First value to compare (current value).
            value_b: Second value to compare (new value).

        Returns:
            True if the second value is lower than the first value, false
            otherwise.

         """

        return value_b < value_a
