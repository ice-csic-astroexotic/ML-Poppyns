"""
    Model for a simple linear neural network.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelLinear(ModelBase):
    """
    A linear neural network Model.
    """

    def __init__(
        self, input_shape: np.array = None, num_parameters: int = 1
    ) -> None:
        """
        Linear model initialization.

        Args:
            input_shape (np.array): Shape of the input batch (C x H x W).
            num_parameters (int): Number of parameters to predict.
        """

        super().__init__()

        input_features = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc1 = nn.Linear(input_features, num_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor for the network.

        Returns:
            (torch.Tensor): Output tensor of the network after forwarding all layers.
        """

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        return x
