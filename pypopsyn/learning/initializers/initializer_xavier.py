"""
    Xavier uniform initializer.

    Class for a xavier uniform weight initializer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import torch

from .initializer_base import InitializerBase


class InitializerXavier(InitializerBase):
    """
    Xavier uniform weight initializer class.

    Any instance of this class is a callable function that will initialize a
    given torch module which contains trainable parameters (weights and biases)
    with a Xavier uniform distribution for the weights and a constant value for
    the biases.
    """

    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a module.

        Weights are initialized using Xavier uniform distribution (see
        "Understanding the difficulty of training deep feedforward neural
        networks" - Glorot, X. & Bengio, Y. (2010) in which the values of are
        sampled from a uniform distribution U(-a,a) where:
        a = gain * sqrt(6 / (fan_in + fan_out)).
        Biases are just filled with a constant close-to-zero value (0.01).

        Args:
            m (torch.module): module with parameters to be initialized. Could
              be anything from a linear layer to a convolutional one. Right now,
              only initialization of Linear layers is performed.

        Returns:
            Nothing.
        """

        if type(m) is torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Args:
            None.

        Returns:
            A string which describes the weight initializer for output purposes.
        """
        return "Xavier Uniform weight initializer"
