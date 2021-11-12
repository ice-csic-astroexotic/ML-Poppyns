"""
Kaiming uniform initializer.

Class for a Kaiming uniform weight initializer.

Authors:

    Alberto Garcia Garcia (garciagarcia@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

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

import torch

from .initializer_base import InitializerBase


class InitializerKaiming(InitializerBase):
    """
    Kaiming uniform weight initializer class.

    Any instance of this class is a callable function that will initialize a
    given torch module which contains trainable parameters (weights and biases)
    with a Kaiming uniform distribution for the weights and a constant value for
    the biases.
    """

    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a module.

        Weights are initialized using Kaiming uniform distribution (see "Delving
        deep into rectifiers: Surpassing human-level performance on ImageNet
        classification" - He, K. et al. (2015)) in which the values of are
        sampled from a uniform distribution U(-bound,bound) where:
        bound = gain * sqrt(3 / fan_mode).
        Biases are just filled with a constant close-to-zero value (0.01).

        Args:
            m (torch.module): module with parameters to be initialized. Could
              be anything from a linear layer to a convolutional one. Right now,
              only initialization of Linear layers is performed.

        Returns:
            Nothing.
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in")
            m.bias.data.fill_(0.01)

    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Args:
            None.

        Returns:
            A string which describes the weight initializer for output purposes.
        """
        return "Kaiming Uniform weight initializer"
