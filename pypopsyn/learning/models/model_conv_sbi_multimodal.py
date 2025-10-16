"""
Model for a multimodal convolutional neural network

Authors:

    Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelConvMultimodalSBI(ModelBase):

    """A multimodal convolutional neural network Model"""

    def __init__(
        self,
        input_shape_1: np.array,
        input_shape_2: np.array,
        len_output_layer: int = 1,
    ) -> None:
        """
        Multi-modal CNN Model Initialization.
        This multimodal CNN automatically adapts to the shape of the initial input features.

        Args:
            input_shape_1: Shape of the input batch (C x H x W) for mode 1.
            input_shape_2: Shape of the input batch (C x H x W) for mode 2.
            len_output_layer (int): Length of the latent vector.

        """

        super().__init__()

        # Set up the convolutional filters for the first mode.
        self.conv1_m1 = nn.Conv2d(
            input_shape_1[0], 32, kernel_size=3, padding=1
        )
        self.conv2_m1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Set up the convolutional filters for the second mode.
        self.conv1_m2 = nn.Conv2d(
            input_shape_2[0], 32, kernel_size=3, padding=1
        )
        self.conv2_m2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Create a mock input with the same shape of the real input drawing values from a normal distribution and pass
        # it through the convolution layers in order to save the shape of the input features after the convolution
        # layer and automatically initialize the linear layers with the right shape.
        x_m1 = torch.randn(input_shape_1).view(
            -1, input_shape_1[0], input_shape_1[1], input_shape_1[2]
        )
        x_m2 = torch.randn(input_shape_2).view(
            -1, input_shape_2[0], input_shape_2[1], input_shape_2[2]
        )

        self._to_linear_m1 = 0
        self._to_linear_m2 = 0

        self.convs_m1(x_m1)
        self.convs_m2(x_m2)

        self.combined_fc1 = nn.Linear(
            self._to_linear_m1 + self._to_linear_m2, len_output_layer
        )

    def convs_m1(self, x):
        """
        Convolution and pooling layers forward pass for mode 1.

        Args:
            x: Input tensor for the convolution layers in mode 1.

        Returns:
            Output tensor of the convolution and pooling layers of mode 1.
        """

        x = self.pool(F.relu(self.conv1_m1(x)))
        x = self.pool(F.relu(self.conv2_m1(x)))

        # If the dimension of the flattened input features to the linear layers has not been saved yet, save it.
        if self._to_linear_m1 == 0:
            self._to_linear_m1 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def convs_m2(self, x):
        """
        Convolution and pooling layers forward pass for mode 2.

        Args:
            x: Input tensor for the convolution layers in mode 2.

        Returns:
            Output tensor of the convolution and pooling layers of mode 2.
        """

        x = self.pool(F.relu(self.conv1_m2(x)))
        x = self.pool(F.relu(self.conv2_m2(x)))

        # If the dimension of the flattened input features to the linear layers has not been saved yet, save it.
        if self._to_linear_m2 == 0:
            self._to_linear_m2 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x1, x2):

        """
        Forward pass.

        Args:
            x1: Input tensor for the mode 1 of the network.
            x2: Input tensor for the mode 2 of the network.

        Returns:
            Output tensor of the network after forwarding all layers.
        """

        # Forward pass through mode 1 of the CNN.
        x1 = self.convs_m1(x1)
        x1 = x1.view(-1, self._to_linear_m1)

        # Forward pass through mode 2 of the CNN.
        x2 = self.convs_m2(x2)
        x2 = x2.view(-1, self._to_linear_m2)

        # Combine the outputs from the two modes.
        x_comb = torch.cat((x1, x2), 1)

        # Forward pass in the fully connected layers.
        x_comb = F.relu(self.combined_fc1(x_comb))

        return x_comb
