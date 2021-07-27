""" mean absolute error loss.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torch
import torch.nn as nn

from .loss_base import LossBase


class LossMAE(LossBase):
    """Mean Absolute Error (MAE) loss"""

    def __call__(self, output, target):

        """ Computation of the MAE loss.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Tensor with a MAE loss value for each input pair output-target.

        """
        self.mae = nn.L1Loss()

        loss = self.mae(output, target)
        return loss

    def __str__(self):

        """ String representation for the MAE loss. """

        return "MAE Loss"
