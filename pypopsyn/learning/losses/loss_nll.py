"""
    Negative log-likelihood loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import torch
import torch.nn.functional as F

from .loss_base import LossBase


class LossNLL(LossBase):
    """
    Negative log-likelihood (NLL) loss.
    """

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computation of the negative log-likelihood.

        Args:
            output (torch.Tensor): Network output tensor (predictions).
            target (torch.Tensor): Ground truth tensor (labels).

        Returns:
            (torch.Tensor): Tensor with a NLL loss value for each input pair output-target.
        """
        return F.nll_loss(output, target)

    def __str__(self) -> str:
        """
        String representation for the NLL loss.

        Returns:
            (str): String representation for the NLL loss.
        """

        return "NLL Loss"
