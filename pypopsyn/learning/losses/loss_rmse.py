""" root mean square error loss.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torch
import torch.nn as nn

from .loss_base import LossBase


class LossRMSE(LossBase):
    "Root Mean Square Error (RMSE) loss"

    def __call__(self, output, target):

        """ Computation of the RMSE loss.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Tensor with a RMS loss value for each input pair output-target.

        """
        self.mse = nn.MSELoss()

        loss = torch.sqrt(self.mse(output, target))
        return loss

    def __str__(self):

        """ String representation for the RMSE loss. """

        return "RMSE Loss"
