""" Model for a convolutional neural network

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase


class ModelNN1(ModelBase):

    """ A convolutional neural network Model """

    def __init__(self, num_parameters=1) -> None:

        """ CNN Model Initialization.

        Args:
            num_parameters: Number of parameters to predict.

        """

        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_parameters)

    def forward(self, x):

        """ Forward pass.

        Args:
            x: Input tensor for the network.

        Returns:
            Output tensor of the network after forwarding all layers.

        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
