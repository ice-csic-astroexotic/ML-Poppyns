import numpy as np
import torch

from .initializer_base import InitializerBase


class InitializerUniform(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        return "Uniform weight initializer."
