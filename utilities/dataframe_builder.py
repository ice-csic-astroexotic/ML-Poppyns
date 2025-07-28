"""
    Helper function to create a DataFrame with a MultiIndex header.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""


import pandas as pd


def build_dataframe(
    data_dict: dict, parameters: list, units: list
) -> pd.DataFrame:
    """
    Helper function to create a DataFrame with a MultiIndex header.

    Args:
        data_dict (dict): A dictionary containing the data to be saved in the dataframe.
        parameters (list): A list of parameter names, used as the first level of the MultiIndex header.
        units (list): A list of physical units, used as the second level of the MultiIndex header.

    Returns:
        (pd.DataFrame): A Pandas DataFrame with a MultiIndex header, where columns are
            indexed by parameters and units.
    """

    header = pd.MultiIndex.from_arrays([parameters, units])

    # Force column order to match `parameters`
    df = pd.DataFrame({key: data_dict[key] for key in parameters})
    df.columns = header

    return df
