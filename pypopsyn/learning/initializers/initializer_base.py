"""
    Base initializer.

    This is an abstract class that contains the skeleton for any weight
    initialization scheme. Note that such weight initializer classes instances
    do behave as callable functions.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc

import torch


class InitializerBase:

    """
    Base abstract class for all weight initializers.
    """

    @abc.abstractmethod
    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a torch module.

        Args:
            m (torch.module): module with parameters to be initialized. Could
                be anything from a linear layer to a convolutional one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Returns:
            (str): A string which describes the weight initializer for output purposes.
        """
        raise NotImplementedError
