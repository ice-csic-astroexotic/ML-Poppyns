import numpy as np
import torch

from .initializer_base import InitializerBase


class InitializerUniformRule(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            torch.nn.init.uniform_(m.weight, -y, y)
            torch.nn.init.constant_(m.bias, 0.0)

    def __str__(self):
        return "Uniform Rule weight initializer."
