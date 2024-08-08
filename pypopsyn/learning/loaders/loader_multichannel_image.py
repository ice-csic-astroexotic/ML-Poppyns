"""
    Loader for multichannel 2D map.

    This loader creates a multichannel 2D image for each sample in the dataset by
    stucking together different 2D density maps.
    These images can be loaded as an input of the neural network together with
    the related values of the labels (ground truth).

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torchvision.transforms
from PIL import Image

from .loader_base import LoaderBase


class DatasetMultichannelImage:
    """
    Dataset for a multichannel image input.
    """

    def __init__(
        self,
        file_path: str,
        ignore: list = [],
        ignore_labels: list = [],
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialization or constructor function for the dataset.

        Args:
            file_path (str): Path to the dataset.csv file containing all the
                information on the dataset.
            ignore (list): Indices of the columns of the dataset that
                will be ignored by the loader.
            ignore_labels (list): Indices of the target/label columns in the dataset that
                will be ignored by the loader.
            transform (Optional[Callable]): Transformations to apply to the images.
        """
        self.dataset = pd.read_csv(file_path)
        # Remove the input columns and labels that are to be ignored.
        self.dataset.drop(
            self.dataset.columns[ignore + ignore_labels], axis=1, inplace=True
        )
        self.transform = transform

    def __len__(self) -> int:
        """
        Length of the dataset (number of samples).

        Returns:
            (int): Length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the dataset and extract the images and the corresponding labels.

        Args:
            index (int): Index running along the rows of the dataset.csv file.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple consisting of a multi-channel 2D image
                with shape N x N x channels (where N is the number of entries
                along a row or column of the array in the .npy file) composed by stacking
                all input images specified in the dataset for the requested sample
                and the corresponding labels for the requested sample.
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
                channel = np.array(Image.open(channel_filename))[:, :, 0]
                channels.append(channel)
            # If an input prefix is not found, it is a label (ground truth) then
            # skip to directly stack them later based on the last index in which
            # we found the input prefix.
            else:
                break

            i += 1

        # Stack all input channels.
        image = np.dstack(channels)
        # Fetch all the labels from the last input channel column.
        labels = np.array(self.dataset.iloc[index, i:], dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


class LoaderMultichannelImage(LoaderBase):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        ignored_inputs: list,
        ignored_labels: list,
        num_workers: int = 1,
        shuffle: bool = False,
    ) -> None:
        """
        Data loader for the density maps dataset. The dataset is expected to be
        packed in dataset.csv file.

        Args:
            data_path (string): Path to the dataset.
            batch_size (int): Number of samples per batch.
            ignored_inputs (list): Indices of columns in the dataset to ignore.
            ignored_labels (list): Indices of columns with labels to ignore.
            num_workers (int): Workers to load the data.
            shuffle (bool): Shuffle the samples or not.
        """

        transformation = torchvision.transforms.ToTensor()

        self.data_path = data_path
        self.ignored_inputs = ignored_inputs
        self.ignored_labels = ignored_labels

        self.dataset = DatasetMultichannelImage(
            self.data_path,
            self.ignored_inputs,
            self.ignored_labels,
            transform=transformation,
        )

        super().__init__(self.dataset, batch_size, num_workers, shuffle)
