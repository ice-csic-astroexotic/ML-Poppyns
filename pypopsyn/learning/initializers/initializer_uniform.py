"""
    Uniform initializer.

    Class for a uniform weight initializer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import numpy as np
import torch

from .initializer_base import InitializerBase


class InitializerUniform(InitializerBase):

    """
    Uniform weight initializer class.

    Any instance of this class is a callable function that will initialize a
    given torch module which contains trainable parameters (weights and biases)
    with a uniform distribution for the weights and zero the biases.
    """

    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a module.

        Weights are initialized using an uniform distribution (with values taken
        from the U(0,1) distribution) whilst biases are just filled with a
        constant zero value.

        Args:
            m (torch.module): Module with parameters to be initialized. Could
                be anything from a linear layer to a convolutional one. Right now,
                only initialization of Linear layers is performed.
        """

        if type(m) is torch.nn.Linear:
            torch.nn.init.uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Returns:
            (str): A string which describes the weight initializer for output purposes.
        """

        return "Uniform weight initializer."
