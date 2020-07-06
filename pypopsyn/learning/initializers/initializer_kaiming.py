import torch

from .initializer_base import InitializerBase


class InitializerKaiming(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in")
            m.bias.data.fill_(0.01)

    def __str__(self):
        return "Kaiming Uniform weight initializer"
