"""
Tests for the basics/cdf_calculator module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

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
import pytest

import pypopsyn.simulator.basics.random_sampler as rs
import pypopsyn.simulator.stellar_dynamics.initial_position as ip

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
        return 0.5

    monkeypatch.setattr(np.random, "uniform", mock_cdf_rand)

    x_rand_out = rs.random_from_pdf(
        test_case_2["x"], pdf, test_case_2["num_draw"]
    )

    assert np.abs(test_case_2["x_rand_expected"] - x_rand_out) < TOL
