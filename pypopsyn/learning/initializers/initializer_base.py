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

    This class serves as a blueprint for creating various initialization methods.
    It defines the essential methods that all weight initializers must implement, ensuring consistency.
    """

    @abc.abstractmethod
    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a torch module.

        Args:
            m (torch.module): Module with parameters to be initialized. Could
                be anything from a linear layer to a convolutional one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        String representation of the weight initializer.

        This method should provide a human-readable description of the weight initializer,
        including its name and any relevant parameters or characteristics.

        Returns:
            (str): String representation of the weight initializer.
        """
        raise NotImplementedError
