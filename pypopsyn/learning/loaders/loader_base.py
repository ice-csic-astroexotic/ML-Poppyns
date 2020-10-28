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


class LoaderBase:
    """
    Base loader abstract class.

    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        num_workers: int,
        collate_fn=torch.utils.data.dataloader.default_collate,
    ):
        """
        Initialization of base loader.

        Args:
            dataset: dataset of maps and labels to load.
            batch_size (int): number of samples per batch.
            num_workers (int): Number of workers (threads) to read data.
            collate_fn: Function to process the list of samples to pack a batch.

        Returns:
            Nothing.
        """

        # Initialize base loader with the provided arguments.
        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        self.train_loader = torch.utils.data.DataLoader(**self.init_kwargs)
        self.valid_loader = torch.utils.data.DataLoader(**self.init_kwargs)
