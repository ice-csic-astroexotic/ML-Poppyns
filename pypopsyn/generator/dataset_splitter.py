""" Splitter for dataset.

    This module splits the provided dataset into two sub-datasets according to a given split fraction.

    Running the code:

        python3 dataset_splitter.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC) 2022

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

import logging
import sys
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)


def split_dataset(dataset_dict: dict, split: float) -> Tuple[dict, dict]:
    """
    This method splits the provided dataset into two sub-datasets set_1 and set_2
    according to a specified split fraction.

    Args:
        dataset_dict (dict): Dictionary containing the information on the dataset.

        split (float): Fraction of set_1 size with respect to the provided dataset size.

    Return:
        (dict, dict): Two dictionaries providing the information on the two sub-datasets created from the split.

    """

    # Check if the split argument falls in the range [0, 1].
    if (split <= 0.0) or (split >= 1.0):
        log.error(
            f"Split argument {split} out of range. It must be in the range (0, 1)."
        )
        sys.exit()

    dataset_size = [len(x) for x in dataset_dict.values()][0]

    # Evaluate the subset1 size according to the fraction defined by the split argument.
    # Then randomly sample subset1 and subset2 from the whole dataset.
    set1_size = int(split * dataset_size)
    dataset_idx = np.arange(dataset_size)

    set1_idx = np.random.choice(dataset_size, set1_size, replace=False)
    set2_idx = np.array([idx for idx in dataset_idx if idx not in set1_idx])

    # Create dictionaries for both datasets.
    set1_dataset_dictionary = {}
    set2_dataset_dictionary = {}

    for k in dataset_dict.keys():
        v = np.array(dataset_dict[k])
        v_set1 = v[set1_idx]
        v_set2 = v[set2_idx]
        set1_dataset_dictionary.setdefault(k, v_set1)
        set2_dataset_dictionary.setdefault(k, v_set2)

    return set1_dataset_dictionary, set2_dataset_dictionary
