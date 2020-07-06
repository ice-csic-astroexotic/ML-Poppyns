import torch

from .initializer_base import InitializerBase


class InitializerXavier(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __str__(self):
        return "Xavier Uniform weight initializer"
