import numpy as np
import pytest

import pypopsyn.simulator.initial_velocity as iv

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "v": 300,
        "p_v_expected": 0.00101,
    }

    return data


def test_pdf_proper_velocity(test_case_1):
    """
    Verifying that proper velocity distribution is correctly calculated.
    """
    p_v_out = iv.pdf_proper_velocity(test_case_1["v"])
    assert np.abs(test_case_1["p_v_expected"] - p_v_out) < TOL
