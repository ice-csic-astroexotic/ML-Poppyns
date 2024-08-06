"""
    Base loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc

import torch


class LossBase:
    """Base abstract class for all losses."""

    @abc.abstractmethod
    def __call__(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Actual computation of the loss function."""

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """String representation of the model."""

        raise NotImplementedError
