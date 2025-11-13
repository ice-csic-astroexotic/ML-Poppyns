"""
    Tests for the initial_velocity module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
            Michele Ronchi (ronchi @ ice.csic.es)
"""


import numpy as np
import pytest

import mlpoppyns.simulator.stellar_dynamics.initial_velocity as iv
import utilities.samplers.random_sampler as rs
from mlpoppyns.simulator.config_simulator import cfg

TOL = 1e-5

# Set the galactic model from Marchetti et al. (2019) for the test.
cfg["galactic_model"] = "gmM19"
# Set the characteristic kick velocity of the exponential model for the test.
cfg["vk_c"] = 180.0
# Set the kick velocity dispersion of the Maxwell model for the test.
cfg["sigma_k"] = 265.0
# Set the parameters of the double Maxwell model for the test.
cfg["sigma_k_comp1"]: float = 55.0
cfg["sigma_k_comp2"]: float = 334.0
cfg["kick_weight_comp1"]: float = 0.19
# Set the maximum kick velocity magnitude in [km/s] for the test .
cfg["vk_extent"]: float = 2500.0


@pytest.fixture()
def test_case_1():

    import mlpoppyns.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {
        "v": np.array([300]),
        "pdf_vk_exp_expected": np.array([0.001749]),
        "pdf_vk_maxwell_expected": np.array([0.002033]),
        "pdf_vk_double_maxwell_expected": np.array([0.001043]),
    }

    return data


@pytest.fixture()
def test_case_2():

    import mlpoppyns.simulator.stellar_dynamics.galactic_model as gm

    gm.initialize_galactic_model()

    data = {"r": 1.0, "z": 1.0, "v_circular_expected": 1.12376e-7}

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "NS_number": 2,
        "kick_rand_expected": np.array(
            [1.0220121572131332e-07, 2.0440243144262663e-07]
        ),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "NS_number": 2,
        "mean": 4.0,
        "sigma": 1.0,
        "kick_rand_expected": np.array(
            [1.5168005289324023e-07, 4.123091315194017e-07]
        ),
    }

    return data


def test_pdf_kick_velocity_exp(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the exponential function.
    """
    pdf_vk_out = iv.pdf_kick_velocity_exp(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_exp_expected"] - pdf_vk_out) < TOL


def test_pdf_kick_velocity_maxwell(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the Maxwell distribution.
    """
    pdf_vk_out = iv.pdf_kick_velocity_maxwell(test_case_1["v"])
    assert np.abs(test_case_1["pdf_vk_maxwell_expected"] - pdf_vk_out) < TOL


def test_pdf_kick_velocity_double_maxwell(test_case_1):
    """
    Verifying that the proper velocity distribution is correctly calculated
    for the double Maxwell distribution.
    """
    pdf_vk_out = iv.pdf_kick_velocity_double_maxwell(test_case_1["v"])
    assert (
        np.abs(test_case_1["pdf_vk_double_maxwell_expected"] - pdf_vk_out)
        < TOL
    )

    # Test if the error is properly raised when the weight is below 0.
    cfg["kick_weight_comp1"] = -0.1
    with pytest.raises(
        ValueError,
        match="The relative weight parameter of the km_double_maxwell kick velocity model must be in "
        "the range 0 and 1.",
    ):
        iv.pdf_kick_velocity_double_maxwell(test_case_1["v"])

    # Test if the error is properly raised when the weight is above 1.
    cfg["kick_weight_comp1"] = 1.1
    with pytest.raises(
        ValueError,
        match="The relative weight parameter of the km_double_maxwell kick velocity model must be in "
        "the range 0 and 1.",
    ):
        iv.pdf_kick_velocity_double_maxwell(test_case_1["v"])


def test_kick_velocity_exp(test_case_3, monkeypatch):
    """
    Verifying that a random velocity is correctly drawn for the Maxwell distribution pdf.
    """
    # Mock a pdf function.
    def mock_pdf(x):
        return np.ones_like(x)

    monkeypatch.setattr(iv, "pdf_kick_velocity_exp", mock_pdf)

    # Mock random_from_pdf to return predictable values.
    def mock_random_from_pdf(vk_grid, pdf_func, ns_number):
        return np.array([100.0, 200.0])  # in km/s

    monkeypatch.setattr(rs, "random_from_pdf", mock_random_from_pdf)

    kick_rand_out = iv.kick_velocity_exp(test_case_3["NS_number"])

    np.testing.assert_allclose(
        kick_rand_out, test_case_3["kick_rand_expected"]
    )

    # Check type and shape
    assert isinstance(kick_rand_out, np.ndarray)
    assert kick_rand_out.shape == (test_case_3["NS_number"],)


def test_kick_velocity_maxwell(test_case_3, monkeypatch):
    """
    Verifying that a random velocity is correctly drawn for the Maxwell distribution pdf.
    """
    # Mock a pdf function.
    def mock_pdf(x):
        return np.ones_like(x)

    monkeypatch.setattr(iv, "pdf_kick_velocity_maxwell", mock_pdf)

    # Mock random_from_pdf to return predictable values.
    def mock_random_from_pdf(vk_grid, pdf_func, ns_number):
        return np.array([100.0, 200.0])  # in km/s

    monkeypatch.setattr(rs, "random_from_pdf", mock_random_from_pdf)

    kick_rand_out = iv.kick_velocity_maxwell(test_case_3["NS_number"])

    np.testing.assert_allclose(
        kick_rand_out, test_case_3["kick_rand_expected"]
    )

    # Check type and shape
    assert isinstance(kick_rand_out, np.ndarray)
    assert kick_rand_out.shape == (test_case_3["NS_number"],)


def test_kick_velocity_double_maxwell(test_case_3, monkeypatch):
    """
    Verifying that a random velocity is correctly drawn for the Maxwell distribution pdf.
    """
    # Mock a pdf function.
    def mock_pdf(x):
        return np.ones_like(x)

    monkeypatch.setattr(iv, "pdf_kick_velocity_double_maxwell", mock_pdf)

    # Mock random_from_pdf to return predictable values.
    def mock_random_from_pdf(vk_grid, pdf_func, ns_number):
        return np.array([100.0, 200.0])  # in km/s

    monkeypatch.setattr(rs, "random_from_pdf", mock_random_from_pdf)

    kick_rand_out = iv.kick_velocity_double_maxwell(test_case_3["NS_number"])

    np.testing.assert_allclose(
        kick_rand_out, test_case_3["kick_rand_expected"]
    )

    # Check type and shape
    assert isinstance(kick_rand_out, np.ndarray)
    assert kick_rand_out.shape == (test_case_3["NS_number"],)


def test_kick_velocity_lognormal(test_case_4, monkeypatch):
    """
    Verifying that a random velocity is correctly drawn for the Maxwell distribution pdf.
    """

    # Mock np.random.normal to return predictable values.
    def mock_random_from_normal(vk_grid, pdf_func, ns_number):
        return np.array([5.0, 6.0])

    monkeypatch.setattr(np.random, "normal", mock_random_from_normal)

    kick_rand_out = iv.kick_velocity_lognormal(
        test_case_4["mean"], test_case_4["sigma"], test_case_4["NS_number"]
    )

    np.testing.assert_allclose(
        kick_rand_out, test_case_4["kick_rand_expected"]
    )

    # Check type and shape
    assert isinstance(kick_rand_out, np.ndarray)
    assert kick_rand_out.shape == (test_case_4["NS_number"],)


def test_circular_velocity(test_case_2):
    """
    Verifying that the circular velocity is evaluated correctly.
    """
    v_circular_out = iv.circular_velocity(test_case_2["r"], test_case_2["z"])

    assert np.isclose(
        v_circular_out,
        test_case_2["v_circular_expected"],
        rtol=TOL,
        atol=1.0e-30,
    )
