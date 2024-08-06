"""
    Mean absolute error loss.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import torch
import torch.nn as nn

from .loss_base import LossBase


class LossMAE(LossBase):
    """
    Mean Absolute Error (MAE) loss.
    """

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computation of the MAE loss.

        Args:
            output (torch.Tensor): Network output tensor (predictions).
            target (torch.Tensor): Ground truth tensor (labels).

        Returns:
            (torch.Tensor): Tensor with a MAE loss value for each input pair output-target.
        """
        self.mae = nn.L1Loss()

        loss = self.mae(output, target)
        return loss

    def __str__(self) -> str:
        """
        String representation for the MAE loss.

        Returns:
            (str): String representation for the MAE loss.
        """

        return "MAE Loss"
