""" Loader for the NN1 model

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

            transform: transformation to apply to the images
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
            Read the dataset and extract the images and the corresponding labels

        Args:
            index (int): index running along the raws of the dataset.csv file

        Returns:
            np.ndarray or torch tensor: multidimensional matrices for the images of
            shape N x N x channels where N is the number of pixels along a raw or
            column of the .png file

            np.ndarray: labels of each image
        """
        channel1_name = self.dataset.iloc[index, 0]
        channel2_name = self.dataset.iloc[index, 2]
        channel3_name = self.dataset.iloc[index, 3]
        channel4_name = self.dataset.iloc[index, 4]

        channel1 = np.array(Image.open(channel1_name))[:, :, 0]
        channel2 = np.array(Image.open(channel2_name))[:, :, 0]
        channel3 = np.array(Image.open(channel3_name))[:, :, 0]
        channel4 = np.array(Image.open(channel4_name))[:, :, 0]

        image = np.dstack((channel1, channel2, channel3, channel4))

        labels = np.array(self.dataset.iloc[index, 5:], dtype=np.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


class LoaderMultichannelImage(LoaderBase):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = False,
    ):
        """
        data loader for the density maps dataset. The dataset is expected to be
        packed in dataset.csv file.

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
