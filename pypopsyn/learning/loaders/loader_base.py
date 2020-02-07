""" Base Loader.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.sampler


class LoaderBase(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=torch.utils.data.dataloader.default_collate,
    ):

        self.shuffle = shuffle
        self.validation_split = validation_split

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.validation_sampler = self._split_sampler(
            self.validation_split
        )

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_validation = int(self.n_samples * split)

        validation_idx = idx_full[0:len_validation]
        train_idx = np.delete(idx_full, np.arange(0, len_validation))

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            validation_idx
        )

        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, validation_sampler
