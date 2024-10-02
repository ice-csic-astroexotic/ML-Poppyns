"""
    JSON utils.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

import collections
import json
import pathlib


def read_json(filename: str) -> dict:
    """
    Read a specified JSON file and generate an ordered dictionary.

    Args:
        filename (str): File path to the JSON file.

    Returns:
        (dict): A dictionary containing the JSON information.
    """

    json_filename = pathlib.Path(filename)
    with json_filename.open("rt") as handle:
        return json.load(handle, object_hook=collections.OrderedDict)
