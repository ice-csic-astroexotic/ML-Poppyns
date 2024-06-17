"""
    Test for the coverage_probability module

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""


import pathlib

import numpy as np
import pytest

from utilities.coverage_probability import coverage_prob


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
