""" Model for a simple linear neural network.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelLinear(ModelBase):

    """ A linear neural network Model """

    def __init__(
        self,
        input_shape: np.array = None,
        num_parameters: int = 1,
        positive: bool = True,
    ) -> None:

        """ Linear model initialization.

        Args:
            input_shape: Shape of the input batch (C x H x W).
            num_parameters: Number of parameters to predict.
            positive: Whether to restrict the output to be positive or not.

        """

        super().__init__()

        input_features = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc1 = nn.Linear(input_features, num_parameters)

        self.positive = positive

    def forward(self, x):

        """ Forward pass.

        Args:
            x: Input tensor for the network.

        Returns:
            Output tensor of the network after forwarding all layers.

        """

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        if self.positive:
            x = torch.sigmoid(x)

        return x
