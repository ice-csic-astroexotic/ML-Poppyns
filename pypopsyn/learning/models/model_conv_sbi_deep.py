"""
Model for a convolutional neural network

Authors:

    Michele Ronchi (ronchi@ice.csic.es)
    Alberto Garcia Garcia (garciagarcia@ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelConvSBIdeep(ModelBase):

    """A convolutional neural network Model"""

    def __init__(
        self, input_shape: np.array, len_output_layer: int = 1
    ) -> None:

        """
        CNN Model Initialization.
        This CNN automatically adapts to the shape of the initial input features.

        Args:
            input_shape: Shape of the input batch (C x H x W).
            num_parameters: Number of parameters to predict.

        """

        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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

    def convs(self, x):
        """
        Convolution and pooling layers forward pass.

        Args:
            x: Input tensor for the convolution layers.

        Returns:
            Output tensor of the convolution and pooling layers.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # If the dimension of the flattened input features to the linear layers has not been saved yet, save it.
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):

        """
        Forward pass.

        Args:
            x: Input tensor for the network.

        Returns:
            Output tensor of the network after forwarding all layers.

        """

        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))

        return x
