import numpy as np
import pytest

import pypopsyn.simulator.cdf_calculator as cc
import pypopsyn.simulator.initial_position as ip


@pytest.fixture()
def test_case_1():
    data = {
        "x": np.array([0.0, 0.1, 0.3, 0.4]),
        "cdf_expected": np.array([0.0, 0.60224, 0.97002, 1]),
    }

    return data


def test_cdf_calculator(test_case_1):
    """
    Checking that the cdf is correctly calculated for given pdf and x array.
    """
    cdf_out = cc.cdf_calculator(test_case_1["x"], ip.pdf_initial_height)
    assert np.isclose(cdf_out, test_case_1["cdf_expected"]).all()
