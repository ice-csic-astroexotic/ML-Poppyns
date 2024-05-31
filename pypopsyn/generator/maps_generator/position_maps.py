"""
Position maps generation routines.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)

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

import logging
import typing

import numpy as np

import pypopsyn.generator.maps_generator.maps2d_generator as mg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
position_map_generators = {
    "array": mg.generate_density_matrix,
    "image": mg.generate_density_map,
}
# Set the corresponding extensions for the types of position maps.
extensions = {"array": "npy", "image": "png"}

log = logging.getLogger(__name__)


def generate_position_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    x_positions: np.array,
    y_positions: np.array,
    x_resolution: int,
    y_resolution: int,
    position_maps_dictionary: dict,
    x_limits: typing.Tuple[float, float] = (-20.0, 20.0),
    y_limits: typing.Tuple[float, float] = (-20.0, 20.0),
):
    """
    This method generates a discrete position map with great flexibility, the
    dimensions of the map can be chosen, the type (image or array) can also be
    decided, and the limits and resolution for it can be specified. As a result,
    a map with the specified filename and an extension determined by the chosen
    type is created as output.

    The dictionary of position maps for the dataset is also updated with the
    generated example.

    Args:
        dataset_path (str): path to the folder where the map will be created.
        map_name (str): specific name for this map.
        sample_number (int): number to suffix this map in the dataset.
        map_type (str): type of map to generate (array or image).
        x_positions (np.array): positions in the first axis (horizontal).
        y_positions (np.array): positions in the second axis (vertical).
        x_resolution (int): resolution in the horizontal axis.
        y_resolution (int): resolution in the vertical axis.
        position_maps_dictionary (dict): partial dictionary of position maps.
        x_limits (float, float): limits of the horizontal axis.
        y_limits (float, float): limits of the vertical axis.

    Returns:
        Nothing.

    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    position_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    position_map_generators[map_type](
        x_positions,
        x_limits,
        y_positions,
        y_limits,
        position_map_filename,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
    )

    # Save density map file names into the partial dataset dictionary.
    position_maps_dictionary.setdefault("input:" + map_name, []).append(
        position_map_filename
    )

    log.info("{} generated...".format(position_map_filename))
