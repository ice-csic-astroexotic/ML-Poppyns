"""
    Model for a shallow convolutional neural network used as an embedding network in the sbi framework to compress the
    input features into a latent vector.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelConvSBIshallow(ModelBase):
    """
    A convolutional neural network model with 1 convolutional filters.
    """

    def __init__(
        self, input_shape: np.array, len_output_layer: int = 1
    ) -> None:
        """
        CNN model initialization.

        This CNN automatically adapts to the shape of the initial input features.

        Args:
            input_shape (np.array): Shape of the input batch (C x H x W).
            len_output_layer (int): Length of the latent vector.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Create a mock input with the same shape of the real input drawing values from a normal distribution and pass
        # it through the convolution layers in order to save the shape of the input features after the convolution
        # layer and automatically initialize the linear layers with the right shape.
        x = torch.randn(input_shape).view(
            -1, input_shape[0], input_shape[1], input_shape[2]
        )
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, len_output_layer)

    def convs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolution and pooling layers forward pass.

        Args:
            x (torch.Tensor): Input tensor for the convolution layers.

        Returns:
            (torch.Tensor): Output tensor of the convolution and pooling layers.
        """

        x = self.pool(F.relu(self.conv1(x)))

        # If the dimension of the flattened input features to the linear layers has not been saved yet, save it.
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor for the network.

        Returns:
            (torch.Tensor): Output tensor of the network after forwarding all layers.
        """

        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))

        return x
