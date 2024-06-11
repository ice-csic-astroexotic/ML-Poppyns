"""
    Base Model.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc

import numpy as np
import torch.nn as nn


class ModelBase(nn.Module):

    """Base abstract class for all models."""

    @abc.abstractmethod
    def forward(self, *inputs):
        """Forward pass

        Abstract method for the forward pass that must be implemented for each
        model that derives this class to implement its whole forward pass.

        Args:
            inputs: The network inputs.

        Returns:
            The network output tensor after forwarding all layers.

        """
        raise NotImplementedError

    def __str__(self):

        """String representation of the model."""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
