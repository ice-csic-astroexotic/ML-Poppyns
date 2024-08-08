"""
    Base loss.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import abc

import torch


class LossBase:
    """
    Base abstract class for all losses.

    This class serves as a blueprint for creating various loss functions
    used in machine learning models. It defines the essential interface
    that all loss functions must implement, ensuring consistency.
    """

    @abc.abstractmethod
    def __call__(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss between the model output and the target.

        This method takes the predicted output from the model and the
        corresponding target values, and computes the loss. The specific
        implementation of this method will vary depending on the type of
        loss function being implemented (e.g., Mean Squared Error, Cross-Entropy).

        Args:
            output (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth values to compare against.

        Returns:
            (torch.Tensor): The computed loss value, which is a scalar tensor
            representing the difference between the output and the target.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        String representation of the loss.

        This method should provide a human-readable description of the
        loss function, including its name and any relevant parameters or
        characteristics.

        Returns:
            (str): A string that describes the loss function.
        """

        raise NotImplementedError
