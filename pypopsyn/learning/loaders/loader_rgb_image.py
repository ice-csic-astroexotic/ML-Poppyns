"""
    Loader for RGB density map images.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torchvision.transforms
from PIL import Image

from .loader_base import LoaderBase


class DatasetRGBImage:
    """
    Upload the images dataset and their labels.
    """

    def __init__(self, file_path: str, transform=None):
        """
        Load the images and labels dataset.

        Args:
            file_path (str): path to the dataset.csv file containing all the
                information on the dataset.
            transform: transformation to apply to the images.
        """
        self.dataset = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self) -> int:
        """
        Length of the dataset (number of samples).

        Returns:
            (int): length of the dataset

        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the dataset and extract the images and the corresponding labels.

        Args:
            index (int): index running along the raws of the dataset.csv file.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple composed by a multidimensional matrices for the images of
                shape N x N x 3 (where N is the number of pixels along a raw or column of
                the .png file) and an array of labels of each image.
        """
        image_name = self.dataset.iloc[index, 0]

        image = np.array(Image.open(image_name))[:, :, 0:3]

        labels = np.array(self.dataset.iloc[index, 1:], dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


class LoaderRGBImage(LoaderBase):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        ignored_inputs: list = [],
        num_workers: int = 1,
        shuffle: bool = False,
    ) -> None:
        """
        Data loader for RGB density maps dataset. The dataset is expected to be
        packed in dataset.csv file.

        Args:
            data_path (string): path to the dataset.
            batch_size (int): Number of samples per batch.
            ignored_inputs (list): Indices of columns in the dataset to ignore.
            num_workers (int): Workers to load the data.
            shuffle (bool): Shuffle the samples or not.
        """

        transformation = torchvision.transforms.ToTensor()

        self.data_path = data_path
        # No possiblity to ignore inputs is given in this dataset. The parameter
        # is just kept for interface purposes.
        self.ignored_inputs = []
        self.dataset = DatasetRGBImage(
            self.data_path, transform=transformation
        )

        super().__init__(self.dataset, batch_size, num_workers, shuffle)
