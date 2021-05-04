""" Multi-Bin loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torch
import torch.nn as nn

from .loss_base import LossBase


class LossMultiBin(LossBase):
    "Multi-Bin loss"

    def __call__(
        self,
        output_class,
        output_residual,
        target_class,
        target_residual,
        eps=1e-6,
        w=1.0,
    ):

        """ Computation of the Multi-Bin loss.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Tensor with a RMS loss value for each input pair output-target.

        """
        self.residual_loss_fn = nn.MSELoss()
        self.class_loss_fn = nn.CrossEntropyLoss()
        # adding a small epsilon to avoid null values

        class_loss: torch.Tensor = self.class_loss_fn(
            output_class, target_class
        )
        residual_loss: torch.Tensor = self.residual_loss_fn(
            output_residual, target_residual
        ) + eps
        return class_loss + w * residual_loss

    def __str__(self):

        """ String representation for the Multi-Bin loss. """

        return "Multi-Bin Loss"
