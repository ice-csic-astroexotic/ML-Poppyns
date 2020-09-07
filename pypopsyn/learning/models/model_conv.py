""" Model for a convolutional neural network

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelConv(ModelBase):

    """ A convolutional neural network Model """

    def __init__(self, input_shape: np.array, num_parameters: int = 1) -> None:

        """ CNN Model Initialization.

        Args:
            input_shape: Shape of the input batch (C x H x W).
            num_parameters: Number of parameters to predict.

        """

        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # TODO: This will need to adapt to different image sizes.
        x = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, num_parameters)

    def convs(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):

        """ Forward pass.

        Args:
            x: Input tensor for the network.

        Returns:
            Output tensor of the network after forwarding all layers.

        """

        self.convs(x)
        # TODO: This will need to adapt to different image sizes.
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))

        return x
