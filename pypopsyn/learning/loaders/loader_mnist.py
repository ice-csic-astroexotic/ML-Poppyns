""" MNIST Loader.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import torchvision.datasets
import torchvision.transforms

from .loader_base import LoaderBase


class LoaderMNIST(LoaderBase):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        transformations = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.data_dir = data_dir
        self.dataset = torchvision.datasets.MNIST(
            self.data_dir,
            train=training,
            download=True,
            transform=transformations,
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
