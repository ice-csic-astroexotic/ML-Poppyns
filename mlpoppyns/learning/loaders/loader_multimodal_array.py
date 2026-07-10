"""
    Loader for multichannel 2D arrays datasets for multimodal training.

    This loader imports the statistics to perform normalization or standardization
    on the targets from an already existent statistics.json file.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import json
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torchvision.transforms

from .loader_base import LoaderBase


class DatasetMultimodalArray:
    """
    Dataset for a multichannel and multimodal array input.

    This class represents a dataset of populations whose representation for any
    of the inputs is a NumPy array of numerical values stored in .npy format. All
    those inputs will be treated as individual channels to generate an input
    tensor for the loader. Labels will be generated as a vector.
    """

    def __import_statistics(self, statistic_path: str) -> None:
        """
        Import dataset statistics for normalization and standardization.

        This routine imports the training dataset statistics that might be needed for
        input/targets normalization and standardization like mean, standard
        deviation, minimum and maximum.

        Args:
            statistic_path (str): Path to the statistics.json file containing the statistics
                of the training dataset.
        """

        # Load statistics from JSON file.
        with open(statistic_path) as read_file:
            self.statistics = json.load(read_file)

        label_name = []

        # Loop over every input column of the dataset to collect all outputs.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset.csv file. Find them and skip them to find the targets names.
            if "input:" not in col:
                label_name.append(col)

        # Save the statistics for the filtered labels.
        self.target_mean = np.array(
            [self.statistics[key]["mean"] for key in label_name],
            dtype=np.float32,
        )
        self.target_std = np.array(
            [self.statistics[key]["std"] for key in label_name],
            dtype=np.float32,
        )
        self.target_max = np.array(
            [self.statistics[key]["max"] for key in label_name],
            dtype=np.float32,
        )
        self.target_min = np.array(
            [self.statistics[key]["min"] for key in label_name],
            dtype=np.float32,
        )

    def __fetch_target_names(self) -> None:
        """
        Fetch the names of the targets/labels from the dataset file.
        """

        self.target_names = []
        # Loop over every input column of the dataset to collect all outputs.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset.csv file. Find them and skip them to find the targets.
            if "input:" not in col:
                self.target_names.append(col)

    def __init__(
        self,
        dataset_path: str,
        statistic_path: str,
        filter_channels: Optional[list] = None,
        filter_labels: Optional[list] = None,
        normalize: bool = False,
        standardize: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialization or constructor routine for the dataset.

        Args:
            dataset_path (str): Path to the dataset.csv file containing all the
                information on the dataset.
            statistic_path (str): Path to the statistics.json file containing the statistics
                of the training dataset.
            filter_channels (list): Indices of the input columns of the dataset that
                will be considered by the loader.
            filter_labels (list): Indices of the target/labels columns in the
                dataset that will be considered by the loader.
            normalize (bool): Whether to normalize inputs and targets or not on
                the fly while loading samples.
            standardize (bool): Whether or not to standardize inputs and targets
                on the fly while loading samples.
            transform (Optional[Callable]): Transformations to apply to the arrays.
        """

        self.normalize = normalize
        self.standardize = standardize
        self.transform = transform

        # Load dataset from .csv file.
        self.dataset = pd.read_csv(dataset_path)

        # Remove the input columns and labels that are to be ignored.
        self.dataset = self.dataset.iloc[:, filter_channels + filter_labels]

        # Import dataset statistics needed for standardization or normalization
        # like mean, standard deviation, minimum, maximum...
        self.__import_statistics(statistic_path)
        self.__fetch_target_names()

    def __len__(self) -> int:
        """
        Length of the dataset (number of samples).

        Returns:
            (int): Length of the dataset
        """
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read the dataset and extract the arrays and the corresponding labels.

        Args:
            index (int): Index running along the rows of the dataset.csv file.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple consisting of a two multichannel 2D arrays
                with shape N x N x channels (where N is the number of entries
                along a row or column of the array in the .npy file) composed by stacking
                all input arrays specified in the dataset for the requested sample
                and the corresponding labels for the requested sample.
        """

        channels_1 = []
        channels_2 = []
        i = 0

        # Loop over the input column of the dataset to get all input channels in
        # a list, so we can stack them later. We assume that all columns must be
        # ordered so "input:" columns go first then all the labels.
        for col in self.dataset.columns:
            # All input channel headers are annotated with a prefix "input:" in
            # the dataset.csv file. Find them and add them to the list.
            # If the channel name contains xray add it to the second multichannel
            # input in order to be processed by the second branch of the neural network.
            if "input:" in col:
                channel_filename = self.dataset.iloc[index, i]
                if "xray" in col:
                    channel_2 = np.array(
                        np.load(channel_filename), dtype=np.float32
                    )
                    channels_2.append(channel_2)
                else:
                    channel_1 = np.array(
                        np.load(channel_filename), dtype=np.float32
                    )
                    channels_1.append(channel_1)
            # If an input prefix is not found, it is a label (ground truth) then
            # skip to directly stack them later based on the last index in which
            # we found the input prefix.
            else:
                break

            i += 1

        # Stack all input channels.
        matrix_1 = np.dstack(channels_1)
        matrix_2 = np.dstack(channels_2)
        # Fetch all the labels from the last input channel column.
        targets = np.array(self.dataset.iloc[index, i:], dtype=np.float32)

        # On-the-fly normalization of inputs and labels. Inputs are normalized
        # on a per-sample basis whilst targets are normalized using dataset-wide
        # statistics.
        if self.normalize:
            per_channel_min_1 = np.min(matrix_1, axis=(0, 1), keepdims=True)
            per_channel_max_1 = np.max(matrix_1, axis=(0, 1), keepdims=True)

            per_channel_min_2 = np.min(matrix_2, axis=(0, 1), keepdims=True)
            per_channel_max_2 = np.max(matrix_2, axis=(0, 1), keepdims=True)

            # Identify channels where per_channel_max equals per_channel_min,
            # indicating that all pixels in the matrix have the same value.
            # This implies that no stars were detected in these simulations.
            zero_norm_mask_1 = (
                per_channel_max_1 == per_channel_min_1
            ).squeeze()
            zero_norm_mask_2 = (
                per_channel_max_2 == per_channel_min_2
            ).squeeze()

            if np.count_nonzero(zero_norm_mask_1) != 0:
                # Set the entire matrix to 0 for channels where per_channel_max
                # == per_channel_min to avoid dividing by zero.
                matrix_1[:, :, zero_norm_mask_1] = 0
            else:
                matrix_1 = (matrix_1 - per_channel_min_1) / (
                    per_channel_max_1 - per_channel_min_1
                )

            if np.count_nonzero(zero_norm_mask_2) != 0:
                # Set the entire matrix to 0 for channels where per_channel_max
                # == per_channel_min to avoid dividing by zero.
                matrix_2[:, :, zero_norm_mask_2] = 0
            else:
                matrix_2 = (matrix_2 - per_channel_min_2) / (
                    per_channel_max_2 - per_channel_min_2
                )

            targets = (targets - self.target_min) / (
                self.target_max - self.target_min
            )

        # On-the-fly standardization of inputs/labels. Inputs are standardized
        # on a per-sample basis whilst targets are normalized using dataset-wide
        # statistics.
        elif self.standardize:
            per_channel_std_1 = np.std(matrix_1, axis=(0, 1), keepdims=True)
            per_channel_mean_1 = np.mean(matrix_1, axis=(0, 1), keepdims=True)

            per_channel_std_2 = np.std(matrix_2, axis=(0, 1), keepdims=True)
            per_channel_mean_2 = np.mean(matrix_2, axis=(0, 1), keepdims=True)

            # Check if per_channel_std equal to 0, indicating that all pixels in
            # the matrix have the same value. This implies that no stars were
            # detected in these simulations.
            zero_std_mask_1 = (per_channel_std_1 == 0).squeeze()
            zero_std_mask_2 = (per_channel_std_2 == 0).squeeze()

            if np.count_nonzero(zero_std_mask_1) != 0:
                # Set the entire matrix to -1 for channels where per_channel_std
                # = 0, to avoid dividing by zero.
                matrix_1[:, :, zero_std_mask_1] = -1
            else:
                matrix_1 = (matrix_1 - per_channel_mean_1) / per_channel_std_1

            if np.count_nonzero(zero_std_mask_2) != 0:
                # Set the entire matrix to -1 for channels where per_channel_std
                # = 0, to avoid dividing by zero.
                matrix_2[:, :, zero_std_mask_2] = -1
            else:
                matrix_2 = (matrix_2 - per_channel_mean_2) / per_channel_std_2

            targets = (targets - self.target_mean) / self.target_std

        # Apply all requested transformations to input.
        if self.transform is not None:
            matrix_1 = self.transform(matrix_1)
            matrix_2 = self.transform(matrix_2)

        return matrix_1, matrix_2, targets


class LoaderMultimodalArray(LoaderBase):
    def __init__(
        self,
        dataset_path: str,
        statistic_path: str,
        batch_size: int,
        filter_inputs: list,
        filter_labels: list,
        num_workers: int = 1,
        shuffle: bool = False,
        normalize: bool = False,
        standardize: bool = False,
    ) -> None:
        """
        Data loader for a multichannel and multimodal array-based dataset. The
        dataset is expected to be packed in a dataset.csv file and contain paths
        to .npy files to be loaded.

        Args:
            dataset_path (string): Path to the dataset.
            statistic_path (string): Path to the dataset.
            batch_size (int): Number of samples per batch.
            filter_inputs (list): Indices of columns in the dataset to consider.
            filter_labels (list): Indices of columns with labels to consider.
            num_workers (int): Workers to load the data.
            shuffle (bool): Shuffle the samples or not.
            normalize (bool): Whether to normalize inputs and targets or not.
            standardize (bool): Whether or not to standardize inputs and targets.
        """

        transformation = torchvision.transforms.ToTensor()

        self.dataset_path = dataset_path
        self.statistic_path = statistic_path
        self.filter_inputs = filter_inputs
        self.filter_labels = filter_labels
        self.normalize = normalize
        self.standardize = standardize

        self.dataset = DatasetMultimodalArray(
            self.dataset_path,
            self.statistic_path,
            self.filter_inputs,
            self.filter_labels,
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
