""" Accuracy metric.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torch

from .metric_base import MetricBase


class MetricAccuracy(MetricBase):
    def __call__(self, output, target):

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

    def __str__(self):

        """ String representation for the Accuracy metric. """

        return "Accuracy Metric"
