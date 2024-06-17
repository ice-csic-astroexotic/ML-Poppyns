"""
    Mean square error loss.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import torch
import torch.nn as nn

from .loss_base import LossBase


class LossMSE(LossBase):
    """Mean Square Error (MSE) loss"""

    def __call__(self, output, target):

        """Computation of the MSE loss.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Tensor with a MSE loss value for each input pair output-target.

        """
        self.mse = nn.MSELoss()

        loss = self.mse(output, target)
        return loss

    def __str__(self):

        """String representation for the MSE loss."""

        return "MSE Loss"
