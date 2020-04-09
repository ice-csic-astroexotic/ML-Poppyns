""" Base Loader.

    Base abstract class for any custom data loader.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import numpy as np
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.sampler


class LoaderBase(torch.utils.data.DataLoader):
    """
    Base loader abstract class.

    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool = False,
        collate_fn=torch.utils.data.dataloader.default_collate,
    ):
        """
        Initialization of base loader.

        Args:
            dataset: dataset of images and labels to load.
            batch_size (int): Batch size for the samplers.
            num_workers (int): Number of workers (threads) to read data.
            shuffle (bool): Random shuffle samples or not.
            collate_fn: Function to process the list of samples to pack a batch.

        Returns:
            Nothing.

        """

        self.n_samples = len(dataset)

        train_idx = np.arange(self.n_samples)

        if shuffle:
            self.sampler = torch.utils.data.sampler.SubsetRandomSampler(
                train_idx
            )
        else:
            self.sampler = torch.utils.data.sampler.SequentialSampler(
                train_idx
            )

        # Initialize base loader with the provided arguments.
        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)
