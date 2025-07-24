"""
    Test for the dataframe_builder module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""

import pandas as pd
import pytest

import utilities.dataframe_builder as dfb


@pytest.fixture()
def test_case_9():
    data = {
        "data_dict": {
            "mass": [1.4, 1.3],
            "radius": [10.0, 9.5],
        },
        "parameters": ["mass", "radius"],
        "units": ["[Msun]", "[kpc]"],
        "expected_columns": pd.MultiIndex.from_arrays(
            [
                ["mass", "radius"],
                ["[Msun]", "[kpc]"],
            ]
        ),
        "expected_data": {
            ("mass", "[Msun]"): [1.4, 1.3],
            ("radius", "[kpc]"): [10.0, 9.5],
        },
        "expected_df": pd.DataFrame(
            {
                ("mass", "[Msun]"): [1.4, 1.3],
                ("radius", "[kpc]"): [10.0, 9.5],
            }
        ),
    }

    return data


def test_build_dataframe(test_case_9):
    """
    Check that the function to create a dataframe is properly working.
    """
    result_df = dfb.build_dataframe(
        test_case_9["data_dict"],
        test_case_9["parameters"],
        test_case_9["units"],
    )

    # Assert the result matches the expected output.
    pd.testing.assert_frame_equal(result_df, test_case_9["expected_df"])
