"""
Tests for the stellar_dynamics/spiral_model module.

    Authors:

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

import pypopsyn.simulator.stellar_dynamics.spiral_model as sm

TOL = 1e-5

smFK06 = sm.SpiralModelFK06()
smYMW17 = sm.SpiralModelYMW17()


@pytest.fixture()
def test_case_1():
    data = {
        "arm_number": 5,
        "NS_number": 2,
        "arm_index_rand_expected": np.array([5, 1]),
        "r": np.array([1.5, 3.0]),
        "arm_index": np.array([3, 1]),
        "phi_expected": np.array([-1.69863, 0.93921]),
    }
    return data


@pytest.fixture()
def test_case_2():
    data = {
        "arm_number": 5,
        "NS_number": 2,
        "arm_index_rand_expected": np.array([5, 1]),
        "r": np.array([1.5, 3.0]),
        "arm_index": np.array([3, 1]),
        "phi_expected": np.array([-3.13512, 0.223778]),
    }
    return data


def test_generate_arm_index_smFK06(test_case_1, monkeypatch):
    """
    Verifying that spiral arm indices are correctly random generated.
    """
    # Mocking the arm indices that are otherwise randomly determined;

    def mock_arm_index(*args, **kwargs):
        return np.array([5, 1])

    monkeypatch.setattr(smFK06, "generate_arm_index", mock_arm_index)

    arm_index_rand_out = smFK06.generate_arm_index(
        test_case_1["arm_number"], test_case_1["NS_number"]
    )

    assert np.isclose(
        test_case_1["arm_index_rand_expected"],
        arm_index_rand_out,
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_check_arm_index_smFK06():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    arm_index = np.array([-1, 6])
    with pytest.raises(ValueError, match="Arm index is out of range."):
        smFK06.check_arm_index(arm_index)


def test_calculate_phi_smFK06(test_case_1):
    """
    Verifying that the angular coordinate phi is correctly calculated.
    """

    phi_out = smFK06.calculate_phi(test_case_1["r"], test_case_1["arm_index"])

    assert np.isclose(
        test_case_1["phi_expected"], phi_out, rtol=TOL, atol=1.0e-30
    ).all()


def test_generate_arm_index_smYMW17(test_case_2, monkeypatch):
    """
    Verifying that spiral arm indices are correctly random generated.
    """
    # Mocking the arm indices that are otherwise randomly determined;
    def mock_arm_index(*args, **kwargs):
        return np.array([5, 1])

    monkeypatch.setattr(smYMW17, "generate_arm_index", mock_arm_index)

    arm_index_rand_out = smYMW17.generate_arm_index(
        test_case_2["arm_number"], test_case_2["NS_number"]
    )

    assert np.isclose(
        test_case_2["arm_index_rand_expected"],
        arm_index_rand_out,
        rtol=TOL,
        atol=1.0e-30,
    ).all()


def test_check_arm_index_smYMW17():
    """
    Verifying that a ValueError is raised if the arm index is out of range.
    """
    arm_index = np.array([-1, 6])
    with pytest.raises(ValueError, match="Arm index is out of range."):
        smYMW17.check_arm_index(arm_index)


def test_calculate_phi_smYMW17(test_case_2):
    """
    Verifying that the angular coordinate phi is correctly calculated.
    """

    phi_out = smYMW17.calculate_phi(test_case_2["r"], test_case_2["arm_index"])

    assert np.isclose(
        test_case_2["phi_expected"], phi_out, rtol=TOL, atol=1.0e-30
    ).all()
