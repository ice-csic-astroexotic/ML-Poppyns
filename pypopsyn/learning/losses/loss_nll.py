"""
    Negative log-likelihood loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import torch.nn.functional as F

from .loss_base import LossBase


class LossNLL(LossBase):

    """Negative log-likelihood (NLL) loss."""

    def __call__(
        self,
        output,
        target,
    ):

        """Computation of the negative log-likelihood.

        Args:
            output: Network output tensor (predictions).
            target: Ground truth tensor (labels).

        Returns:
            Tensor with a NLL loss value for each input pair output-target.

        """
        return F.nll_loss(output, target)

    def __str__(self):

        """String representation for the NLL loss."""

        return "NLL Loss"
