"""
    Base model.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc

import numpy as np
import torch
import torch.nn as nn


class ModelBase(nn.Module):
    """
    Base abstract class for all models.

    This class serves as a blueprint for creating various neural-network architecture models.
    It defines the essential methods that all models must implement, ensuring consistency.
    """

    @abc.abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Abstract method for the forward pass that must be implemented for each
        model that derives this class to implement its whole forward pass.

        Args:
            inputs (torch.Tensor): The network inputs.

        Returns:
            (torch.Tensor): The network output tensor after forwarding all layers.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        String representation of the model.

        This method should provide a human-readable description of the model,
        including its name and any relevant parameters or characteristics.

        Returns:
            (str): String representation of the model.
        """

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
