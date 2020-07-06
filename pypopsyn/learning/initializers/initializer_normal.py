import numpy as np
import torch

from .initializer_base import InitializerBase


class InitializerNormal(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            y = m.in_features
            torch.nn.init.normal_(m.weight, 0.0, 1.0 / np.sqrt(y))
            m.bias.data.fill_(0.0)

    def __str__(self):
        return "Normal weight initializer"
