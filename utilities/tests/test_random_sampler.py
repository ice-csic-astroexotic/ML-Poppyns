"""
    Tests for the random_sampler module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import mlpoppyns.simulator.stellar_dynamics.initial_position as ip
import utilities.samplers.random_sampler as rs

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "x": np.array([0.0, 0.1, 0.3, 0.4]),
        "cdf_expected": np.array([0.0, 0.46338, 0.91248, 1.0]),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "x": np.linspace(0.0, 10.0, 5),
        "cdf": 1.0 / 10.0 * np.linspace(0.0, 10.0, 5),
        "num_draw": 1,
        "x_rand_expected": 5.0,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "x1": np.linspace(0.0, 10.0, 5),
        "x2": np.linspace(0.0, 10.0, 5),
        "pdf2D": np.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.5, 0.5, 0.5, 0.0],
                [0.0, 0.5, 1.0, 0.5, 0.0],
                [0.0, 0.5, 0.5, 0.5, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.1],
            ]
        ),
        "num_draw": 1,
        "x1_rand_expected": np.array([5.0]),
        "x2_rand_expected": np.array([5.0]),
    }

    return data


def test_cdf_calculator(test_case_1):
    """
    Checking that the cdf is correctly calculated for given pdf and x array.
    """
    cdf_out = rs.cdf_calculator(test_case_1["x"], ip.pdf_initial_height)
    assert np.isclose(cdf_out, test_case_1["cdf_expected"]).all()


def test_random_from_cdf(monkeypatch, test_case_2):
    """
    Checking that random numbers are correctly drawn from a cdf.
    """

    def mock_cdf_rand(*args, **kwargs):
        return 0.5

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x_rand_out = rs.random_from_cdf(
        test_case_2["x"], test_case_2["cdf"], test_case_2["num_draw"]
    )

    assert np.abs(test_case_2["x_rand_expected"] - x_rand_out) < TOL


def test_random_from_pdf(monkeypatch, test_case_2):
    """
    Checking that random numbers are correctly drawn from a pdf.
    """

    def pdf(x: np.ndarray) -> np.ndarray:
        return np.ones(len(x)) * 1.0 / 10.0

    def mock_cdf_rand(*args, **kwargs):
        return np.array([0.5])

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x_rand_out = rs.random_from_pdf(
        test_case_2["x"], pdf, test_case_2["num_draw"]
    )

    assert np.abs(test_case_2["x_rand_expected"] - x_rand_out) < TOL


def test_random_from_pdf2d(monkeypatch, test_case_3):
    """
    Checking that random numbers are correctly drawn from a 2D pdf.
    """

    def mock_cdf_rand(*args, **kwargs):
        return np.array([0.5])

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x1_rand_out, x2_rand_out = rs.random_from_pdf_2d(
        test_case_3["x1"],
        test_case_3["x2"],
        test_case_3["pdf2D"],
        test_case_3["num_draw"],
    )

    assert np.abs(test_case_3["x1_rand_expected"] - x1_rand_out) < TOL
    assert np.abs(test_case_3["x2_rand_expected"] - x2_rand_out) < TOL
