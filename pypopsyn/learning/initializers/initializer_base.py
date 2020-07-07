"""
Base initializer.

This is an abstract class that contains the skeleton for any weight
initialization scheme. Note that such weight initializer classes instances
do behave as callable functions.

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

import abc

import torch


class InitializerBase:

    """ Base abstract class for all weight initializers. """

    @abc.abstractmethod
    def __call__(self, m: torch.nn.Module) -> None:
        """
        Custom call operator for initializing the parameters of a torch module.

        Args:
            m (torch.module): module with parameters to be initialized. Could
              be anything from a linear layer to a convolutional one.

        Returns:
            Nothing.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Custom to string operator for the weight initializer.

        Args:
            None.

        Returns:
            A string which describes the weight initializer for output purposes.
        """
        raise NotImplementedError
