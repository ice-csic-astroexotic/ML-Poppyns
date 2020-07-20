"""
Loader for multichannel 2D arrays datasets.

Authors:

    Michele Ronchi (ronchi@ice.csic.es)
    Alberto Garcia Garcia (garciagarcia@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image

from .loader_base import LoaderBase


class DatasetMultichannelArray:
    """
    Dataset for a multichannel array input.

    This class represents a dataset of populations whose representation for any
    of the inputs is a numpy array of numerical values stored in NPY format. All
    those inputs will be treated as individual channels to generate an input
    tensor for the loader. Labels will be generated as a vector.
    """

    def __compute_statistics(self):
        """
        Compute dataset statistics for normalization and standardization.

        This routine computes dataset-wide statistics that might be needed for
        input/targets normalization and standardization like mean, standard
        deviation, minimum, maximum...

        Args:
            None.

        Returns:
            Nothing.

        """

        i = 0

        # Loop over every input column of the dataset to collect all outputs.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset CSV file. Find them and skip them to find the targets.
            if "input:" in col:
                i += 1
            # If an input prefix is not found, it is a label (ground truth) then
            # skip to directly stack them later based on the last index in which
            # we found the input prefix.
            else:
                break

        # Fetch all the targets from the last input channel column.
        targets = np.array(self.dataset.iloc[:, i:], dtype=np.float32)

        # Compute statistics for targets. Note that they are computed on a
        # per-position/channel basis over the whole dataset so if we have
        # multiple labels for each sample, we compute the statistics for each
        # one of the labels across the whole set of samples (hence axis=0).
        self.target_std = np.std(targets, axis=0)
        self.target_mean = np.mean(targets, axis=0)
        self.target_max = np.max(targets, axis=0)
        self.target_min = np.min(targets, axis=0)

    def __fetch_target_names(self):
        """
        Fetch the names of the targets/labels from the dataset file.

        Args:
            None.

        Returns:
            Nothing.

        """

        self.target_names = []
        # Loop over every input column of the dataset to collect all outputs.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset CSV file. Find them and skip them to find the targets.
            if "input:" not in col:
                self.target_names.append(col)

    def __init__(
        self,
        file_path,
        ignore=[],
        ignore_labels=[],
        normalize=False,
        standardize=False,
        transform=None,
    ) -> None:
        """
        Initialization or constructor routine for the dataset.

        Args:
            file_path (str): path to the dataset.csv file containing all the
                information on the dataset.
            ignore (list): indices of the input columns of the dataset that
                will be ignored by the loader.
            ignore_labels (list): indices of the target/labels columns in the
                dataset that will be ignored by the loader.
            normalize (bool): whether to normalize inputs and targets or not on
                the fly while loading samples.
            standardize (bool): whether or not to standardize inputs and targets
                on the fly while loading samples.
            transform: transformations to apply to the arrays.

        Returns:
            Nothing.

        """

        self.normalize = normalize
        self.standardize = standardize
        self.transform = transform

        # Load dataset from CSV file.
        self.dataset = pd.read_csv(file_path)

        # Remove the input columns and labels that are to be ignored.
        self.dataset.drop(
            self.dataset.columns[ignore + ignore_labels], axis=1, inplace=True
        )

        # Compute dataset statistics needed for standardization or normalization
        # like mean, standard deviation, minimum, maximum...
        self.__compute_statistics()
        self.__fetch_target_names()

    def __len__(self):
        """
        Length of the dataset (number of samples).

        Returns:
            int: length of the dataset

        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Read the dataset and extract the arrays and the corresponding labels.

        Args:
            index (int): index running along the rows of the dataset CSV file.

        Returns:
            np.ndarray: multi-channel 2D array composed by stacking all input
            arrays specified in the dataset for the requested sample with
            shape N x N x channels where N is the number of entries along a
            row or column of the array in the .npy file.

            np.ndarray: labels for the requested sample.

        """

        channels = []
        i = 0

        # Loop over the input column of the dataset to get all input channels in
        # a list so we can stack them later. We assume that all columns must be
        # ordered so "input:" columns go first then all the labels.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset CSV file. Find them and add them to the list.
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
        targets = np.array(self.dataset.iloc[index, i:], dtype=np.float32)

        # On-the-fly normalization of inputs and labels. Inputs are normalized
        # on a per-sample basis whilst targets are normalized using dataset-wide
        # statistics.
        if self.normalize:
            per_channel_min = np.min(matrix, axis=(0, 1), keepdims=True)
            per_channel_max = np.max(matrix, axis=(0, 1), keepdims=True)
            matrix = (matrix - per_channel_min) / (
                per_channel_max - per_channel_min
            )

            targets = (targets - self.target_min) / (
                self.target_max - self.target_min
            )

        # On-the-fly standardization of inputs/labels. Inputs are standardized
        # on a per-sample basis whilst targets are normalized using dataset-wide
        # statistics.
        elif self.standardize:
            per_channel_std = np.std(matrix, axis=(0, 1), keepdims=True)
            per_channel_mean = np.mean(matrix, axis=(0, 1), keepdims=True)
            matrix = (matrix - per_channel_mean) / per_channel_std

            targets = (targets - self.target_mean) / self.target_std

        # Apply all requested transformations to input.
        if self.transform is not None:
            matrix = self.transform(matrix)

        return matrix, targets


class LoaderMultichannelArray(LoaderBase):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        ignored_inputs: list,
        ignored_labels: list,
        num_workers: int = 1,
        shuffle: bool = False,
        normalize: bool = False,
        standardize: bool = False,
    ):
        """
        Data loader for a multi-channel array-based dataset. The dataset is
        expected to be packed in a dataset.csv file and contain paths to .npy
        files to be loaded.

        Args:
            data_path (string): path to the dataset.
            batch_size (int): Number of samples per batch.
            ignored_inputs (list): Indices of columns in the dataset to ignore.
            ignored_labels (list): Indices of columns with labels to ignore.
            num_workers (int): Workers to load the data.
            shuffle (bool): Shuffle the samples or not.
            normalize (bool): whether to normalize inputs and targets or not.
            standardize (bool): whether or not to standardize inputs and targets.

        Returns:
            Nothing.

        """

        transformation = torchvision.transforms.ToTensor()

        self.data_path = data_path
        self.ignored_inputs = ignored_inputs
        self.ignored_labels = ignored_labels
        self.normalize = normalize
        self.standardize = standardize

        self.dataset = DatasetMultichannelArray(
            self.data_path,
            self.ignored_inputs,
            self.ignored_labels,
            self.normalize,
            self.standardize,
            transform=transformation,
        )

        self.target_mean = self.dataset.target_mean
        self.target_std = self.dataset.target_std
        self.target_max = self.dataset.target_max
        self.target_min = self.dataset.target_min
        self.target_names = self.dataset.target_names

        super().__init__(self.dataset, batch_size, num_workers, shuffle)
