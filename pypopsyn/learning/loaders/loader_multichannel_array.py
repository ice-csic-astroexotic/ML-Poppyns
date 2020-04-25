""" Loader for the multichannel 2D arrays.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import pandas as pd
import torchvision.transforms
from PIL import Image

from .loader_base import LoaderBase


class DatasetUpload:
    """
        upload the images dataset and their labels
    """

    def __init__(self, file_path, transform=None):
        """
            load the images and labels dataset
        Args:
            file_path (str): path to the dataset.csv file containing all the
            information on the dataset

            transform: transformation to apply to the arrays
        """
        self.dataset = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        """

        Returns:
            int = length of the dataset

        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
            Read the dataset and extract the arrays and the corresponding labels

        Args:
            index (int): index running along the raws of the dataset.csv file

        Returns:
            np.ndarray or torch tensor: multi-channel 2D array composed by stacking
            all input arrays specified in the dataset for the requested sample with
            shape N x N x channels where N is the number of entries along a row or
            column of the array in the .npy file.

            np.ndarray: labels of each input.
        """

        channels = []
        i = 0

        # Loop over every input column of the dataset to collect all input channels
        # in a list so we can stack them later. We assume that all columns must be
        # ordered so "input:" columns go first then all the labels.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in the
            # dataset CSV file. Find them and add them to the list.
            if "input:" in col:
                channel_filename = self.dataset.iloc[index, i]
                channel = np.array(np.load(channel_filename), dtype=np.float32)
                channels.append(channel)
            # If an input prefix is not found, it is a label (ground truth) then
            # skip to directly stack them later based on the last index in which
            # we found the input prefix.
            else:
                break

            i += 1

        # Stack all input channels.
        matrix = np.dstack(channels)
        # Fetch all the labels from the last input channel column.
        labels = np.array(self.dataset.iloc[index, i:], dtype=np.float32)

        if self.transform is not None:
            matrix = self.transform(matrix)

        return matrix, labels


class LoaderMultichannelArray(LoaderBase):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = False,
    ):
        """
        data loader for the density maps dataset. The dataset is expected to be
        packed in a dataset.csv file.

        Args:
            data_path (string): path to the dataset.
            batch_size (int): Number of samples per batch.
            num_workers (int): Workers to load the data.
            shuffle (bool): Shuffle the samples or not.

        Returns:
            Nothing

        """

        transformation = torchvision.transforms.ToTensor()

        self.data_path = data_path
        self.dataset = DatasetUpload(self.data_path, transform=transformation)

        super().__init__(self.dataset, batch_size, num_workers, shuffle)
