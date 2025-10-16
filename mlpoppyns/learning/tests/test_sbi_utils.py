"""
 Test for the sbi_utils module.

    Authors:
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

import mlpoppyns.learning.utils.sbi_utils as sbi_utils
from mlpoppyns.learning.utils.sbi_utils import calculate_smallest_hdr


@pytest.fixture
def fake_round_datasets(tmp_path):
    """
    Creates a fake dataset structure like:
    tmp_path/
        round_0/dataset_full.csv
        round_1/dataset_full.csv
        round_2/dataset_full.csv
    """

    # Create dummy data for each round.
    num_rounds = 3
    for i in range(num_rounds):
        round_dir = tmp_path / f"round_{i}"
        round_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "B_initial_log10_mean": np.random.randn(2),
                "B_initial_log10_sigma": np.random.randn(2),
            }
        )

        df.to_csv(round_dir / "dataset_full.csv", index=False)

    return tmp_path, num_rounds - 1


@pytest.fixture
def fake_inference(tmp_path):

    # Mock inference object with _summary_writer.log_dir.
    inference = MagicMock()
    summary_writer = MagicMock()
    summary_writer.log_dir = str(tmp_path / "tensorboard_logs")
    inference._summary_writer = summary_writer
    return inference


@pytest.fixture
def fake_dataset():
    dataset = MagicMock()
    dataset.target_max = [13.99, 0.99]
    dataset.target_min = [12.00, 0.10]
    dataset.target_mean = [13.01, 0.54]
    dataset.target_std = [0.57, 0.26]
    dataset.target_names = ["B_initial_log10_mean", "B_initial_log10_sigma"]
    dataset.normalize = False
    dataset.standardize = True
    return dataset


@pytest.fixture
def fake_posterior():
    mock = MagicMock()

    # Mocking log_prob and set it to -1.
    def log_prob(samples, _):
        if samples.ndim == 1:
            return torch.tensor(-1.0)
        return torch.full((samples.shape[0],), -1.0)

    mock.log_prob.side_effect = log_prob
    return mock


@pytest.fixture
def fake_sampler(monkeypatch):
    # Simulate successful posterior sampling.
    def sample_with_timeout(posterior, x, n_samples, timeout):
        samples = torch.randn((n_samples, x.shape[1]))
        return samples, True

    monkeypatch.setattr(
        sbi_utils.sampler, "sample_with_timeout", sample_with_timeout
    )


@pytest.fixture
def make_config(tmp_path):
    config = MagicMock()
    config.log_dir = str(tmp_path)
    return config


@pytest.fixture
def dummy_logger():
    logger = MagicMock()
    logger.exception = MagicMock()
    logger.error = MagicMock()
    logger.info = MagicMock()
    return logger


def test_calculate_smallest_hdr(fake_posterior, fake_sampler, dummy_logger):
    """
    Test for the calculate_smallest_hdr function.

    Args:
        fake_posterior (MagicMock): Mocked posterior object with a fixed log_prob output.
        fake_sampler (fixture): Fixture that monkeypatches the sampler to return random samples.
        dummy_logger (MagicMock): Logger object that records info messages.
    """
    n_test_samples = 5
    n_param = 3
    n_posterior_samples = 100

    theta = torch.randn((n_test_samples, n_param))
    matrix = torch.randn((n_test_samples, 3, 32, 32))
    device = torch.device("cpu")

    hdr, _ = calculate_smallest_hdr(
        posterior=fake_posterior,
        theta=theta,
        matrix=matrix,
        n_samples=n_posterior_samples,
        logger=dummy_logger,
        device=device,
    )

    assert isinstance(hdr, np.ndarray)
    assert hdr.shape[0] == n_test_samples
    assert np.all((hdr >= 0.0) & (hdr <= 1.0))


def test_corner_plot(tmp_path, fake_dataset):
    """
    Test for the corner_plot function.

    Args:
        tmp_path (Path): Temporary directory where the plot will be saved.
        fake_dataset (MagicMock): Mocked dataset containing parameter stats and metadata.
    """
    # Generate fake posterior samples.
    n_samples = 100
    n_params = 2
    samples = torch.randn((n_samples, n_params))

    output_file = tmp_path / "corner_plot.png"

    sbi_utils.corner_plot(samples, fake_dataset, str(output_file))

    # Check if file was created.
    assert os.path.exists(output_file)

    plt.close("all")


def test_merge_all_rounds_dataset(fake_round_datasets):
    """
    Test for the merge_all_rounds_dataset function.

    Args:
        fake_round_datasets (tuple): Function that create the subfolders structure and the dummy datasets.
    """
    base_path, last_completed_round = fake_round_datasets
    output_path = sbi_utils.merge_all_rounds_dataset(
        base_path, last_completed_round
    )

    # Expected output file.
    merged_file = Path(output_path) / "dataset_full.csv"
    assert merged_file.exists()

    # Check if the len of dataset is correct.
    merged_df = pd.read_csv(merged_file)
    assert len(merged_df) == 6


@patch("sbi.analysis.tensorboard_output._get_event_data_from_log_dir")
def test_save_training_statistics(
    mock_get_event_data, make_config, fake_inference
):
    """
    Test for the save_training_statistics function.

    Args:
        mock_get_event_data (MagicMock): Patched function to return dummy tensorboard scalar data.
        make_config (MagicMock): Configuration object with log_dir attribute.
        fake_inference (MagicMock): Mocked sbi inference object with summary writer.
    """
    # Prepare fake event data.
    fake_event_data = {
        "scalars": {
            "training_log_probs": {
                "step": [1, 2, 3],
                "value": [2, 4, 5],
            },
            "validation_log_probs": {
                "step": [1, 2, 3],
                "value": [2.1, 4.1, 5.1],
            },
        }
    }

    mock_get_event_data.return_value = fake_event_data

    index = 0
    effective_round = 1

    sbi_utils.save_training_statistics(
        make_config, fake_inference, index, effective_round
    )
    round_dir = os.path.join(make_config.log_dir, f"round_{effective_round}")

    json_path = os.path.join(round_dir, f"training_statistics_{index}.json")
    pdf_path = os.path.join(round_dir, f"training_stats_{index}.png")

    # Assert both files are created.
    assert os.path.exists(json_path)
    assert os.path.exists(pdf_path)

    # Load and check JSON structure.
    with open(json_path, "r") as f:
        data = json.load(f)
    assert "training_log_probs" in data
    assert "validation_log_probs" in data

    plt.close("all")


@patch("mlpoppyns.learning.utils.sbi_utils.dl.DatasetMultichannelArray")
def test_prepare_dataset_sbi(dummy_logger):
    """
    Integration test for prepare_dataset_sbi using a real dataset and config file.

    Asserts:
        dummy_logger (MagicMock): Logger object that records info messages.
    """
    dataset_folder = "data/example_generator_magrot"
    config_path = Path("mlpoppyns/learning/config_sbi.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    config["compression_input"]["use_compression"] = False
    config["trainer"]["type"] = "snpe"
    dataset, parameter, matrix = sbi_utils.prepare_dataset_sbi(
        dataset_folder=dataset_folder,
        config=config,
        logger=dummy_logger,
        atnf=False,
    )

    assert parameter.shape[0] == matrix.shape[0] == len(dataset)
    assert matrix.shape[2] == config["arch"]["args"]["input_shape"][1]
