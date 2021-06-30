""" Splitter for dataset.

    This module split the provided dataset into two sub-dataset according to a given split fraction.

    Running the code:

        python3 dataset_splitter.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

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

from typing import Tuple

import numpy as np


def split_dataset(dataset_dict: dict, split: float) -> Tuple[dict, dict]:
    """
    This method splits the provided dataset into two sub-datasets according to a specified split fraction.

    Args:
        dataset_dict (dict): Dictionary containing the information on the all dataset.

        split (float): Fraction of validation/test dataset size with respect to the train dataset.

    Return:
        (dict, dict): Two dictionaries providing the information on the two sub-dataset created from the split.

    """

    dataset_size = [len(x) for x in dataset_dict.values()][0]
    print(dataset_size)

    # Evaluate the validation dataset size according to the fraction defined by the split argument.
    # Then random sample the validation dataset and the train dataset from the whole dataset.
    valid_size = int(split * dataset_size)
    dataset_idx = np.arange(dataset_size)
    valid_idx = np.random.choice(dataset_size, valid_size, replace=False)
    train_idx = np.array([idx for idx in dataset_idx if idx not in valid_idx])
    # Create dictionaries for the training and validation datasets.
    valid_dataset_dictionary = {}
    train_dataset_dictionary = {}
    for k in dataset_dict.keys():
        v = np.array(dataset_dict[k])
        v_valid = v[valid_idx]
        v_train = v[train_idx]
        valid_dataset_dictionary.setdefault(k, v_valid)
        train_dataset_dictionary.setdefault(k, v_train)

    return valid_dataset_dictionary, train_dataset_dictionary
