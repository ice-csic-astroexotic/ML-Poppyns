import numpy as np
import pytest

import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.initial_position as ip

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "x": np.array([0.0, 0.1, 0.3, 0.4]),
        "cdf_expected": np.array([0.0, 0.60224, 0.97002, 1]),
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


def test_cdf_calculator(test_case_1):
    """
    Checking that the cdf is correctly calculated for given pdf and x array.
    """
    cdf_out = cc.cdf_calculator(test_case_1["x"], ip.pdf_initial_height)
    assert np.isclose(cdf_out, test_case_1["cdf_expected"]).all()


def test_random_from_cdf(monkeypatch, test_case_2):
    """
    Checking that random numbers are correctly drawn from a cdf
    """

    def mock_cdf_rand(*args, **kwargs):
        return 0.5

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x_rand_out = cc.random_from_cdf(
        test_case_2["x"], test_case_2["cdf"], test_case_2["num_draw"]
    )

    assert np.abs(test_case_2["x_rand_expected"] - x_rand_out) < TOL


def test_random_from_pdf(monkeypatch):
    """
    Checking that random numbers are correctly drawn from a pdf
    """

    def pdf(x: float) -> float:
        return 1.0 / 10.0

    x_grid = np.linspace(0.0, 10.0, 5)

    num_draw = 1

    x_rand_expected = 5.0

    def mock_cdf_rand(*args, **kwargs):
        return 0.5

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x_rand_out = cc.random_from_pdf(x_grid, pdf, num_draw)

    assert np.abs(x_rand_expected - x_rand_out) < TOL
