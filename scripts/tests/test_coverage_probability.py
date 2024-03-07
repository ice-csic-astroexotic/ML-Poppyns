"""
Test for the coverage_probability.py module

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2024

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


import pathlib

import numpy as np
import pytest

from scripts.coverage_probability import coverage_prob


@pytest.fixture
def sample_data():
    # Generating sample data
    hdr_testset = np.random.rand(10000)
    n_betas = 10
    save_dir = pathlib.Path("test_output")
    save_dir.mkdir(exist_ok=True)
    return hdr_testset, n_betas, save_dir


def test_coverage_prob(sample_data):
    """
    Test of the `coverage_prob` function to verify whether the coverage probability numpy array
    and plot are correctly created and saved.
    """

    hdr_testset, n_betas, save_dir = sample_data

    coverage_prob(hdr_testset, n_betas, save_dir)

    # Check if the output files exist.
    assert (save_dir / "coverage_probability.npy").exists()
    assert (save_dir / "coverage_plot.pdf").exists()

    # Check if the coverage_probability numpy array contains correct data.
    coverage_probability = np.load(save_dir / "coverage_probability.npy")
    assert len(coverage_probability) == n_betas
