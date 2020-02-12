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
        data_dir: str,
        batch_size: int,
        num_workers: int = 1,
        training: bool = True,
    ):
        """
        Example data loader for the MNIST dataset.

        Args:
            data_dir: Directory to download the data.
            batch_size: Number of samples per batch.
            num_workers: Workers to load the data.
            training: Load in training mode.

        Returns:
            Nothing

        """

        # Apply MNIST normalization transformation.
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

        super().__init__(self.dataset, batch_size, num_workers)
