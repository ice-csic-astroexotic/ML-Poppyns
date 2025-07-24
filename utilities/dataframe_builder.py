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
    # If the key `"idx"` is present, it is removed.
    data_dict.pop("idx", None)

    header = pd.MultiIndex.from_arrays([parameters, units])

    df = pd.DataFrame.from_dict(data=data_dict)
    df.columns = header

    return df
