"""
    Normal initializer.

    Class for a normal weight initializer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import numpy as np
import torch

from .initializer_base import InitializerBase


class InitializerNormal(InitializerBase):
    """
    Normal weight initializer class.

    Any instance of this class is a callable function that will initialize a
    given torch module which contains trainable parameters (weights and biases)
    with a normal distribution for the weights and zero the biases.
    """

    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a module.

        Weights are initialized using a normal distribution (with values taken
        from the N(mean, std^2) distribution) whilst biases are just filled with
        a constant zero value. In this case, std = 1 / sqrt(y) where y is the
        number of input features.

        Args:
            m (torch.module): module with parameters to be initialized. Could
              be anything from a linear layer to a convolutional one. Right now,
              only initialization of Linear layers is performed.

        Returns:
            Nothing.
        """
        if type(m) is torch.nn.Linear:
            y = m.in_features
            torch.nn.init.normal_(m.weight, 0.0, 1.0 / np.sqrt(y))
            m.bias.data.fill_(0.0)

    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Args:
            None.

        Returns:
            A string which describes the weight initializer for output purposes.
        """

        return "Normal weight initializer"
