import torch

from .initializer_base import InitializerBase


class InitializerDirac(InitializerBase):
    def __call__(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.dirac_(m.weight)
            m.bias.data.fill_(0.0)

    def __str__(self):
        return "Dirac weight initializer"
