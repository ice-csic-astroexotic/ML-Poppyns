"""
P-P_dot maps generation routines.

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

import pypopsyn.generator.dataset_generator as dg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
ppdot_map_generators = {
    "array": dg.generate_density_matrix,
    "image": dg.generate_density_map,
}
# Set the corresponding extensions for the types of position maps.
extensions = {"array": "npy", "image": "png"}

log = logging.getLogger(__name__)


def generate_ppdot_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    periods: np.array,
    period_derivatives: np.array,
    p_resolution: int,
    pdot_resolution: int,
    ppdot_maps_dictionary: dict,
    p_limits: typing.Tuple[float, float] = (0.001, 100.0),
    pdot_limits: typing.Tuple[float, float] = (1.0e-21, 1.0e-9),
):
    """
    This method generates a discrete p-p_dot diagram map with great flexibility, the
    dimensions of the map can be chosen, the type (image or array) can also be
    decided, and the limits and resolution for it can be specified. As a result,
    a map with the specified filename and a extension determined by the chosen
    type is created as output.

    The dictionary of p-p_dot maps for the dataset is also updated with the
    generated example.

    Args:
        dataset_path (str): path to the folder where the map will be created.
        map_name (str): specific name for this map.
        sample_number (int): number to suffix this map in the dataset.
        map_type (str): type of map to generate (array or image).
        periods (np.array): spin periods of the neutron stars (horizontal axis).
        period_derivatives (np.array): spin period derivatives of the neutron stars (vertical axis).
        p_resolution (int): resolution in the horizontal axis.
        pdot_resolution (int): resolution in the vertical axis.
        ppdot_maps_dictionary (dict): partial dictionary of p-p_dot maps.
        p_limits (float, float): limits of the horizontal axis.
        pdot_limits (float, float): limits of the vertical axis.

    Returns:
        Nothing.

    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    ppdot_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    ppdot_map_generators[map_type](
        periods,
        p_limits,
        period_derivatives,
        pdot_limits,
        ppdot_map_filename,
        x_log_scale=True,
        y_log_scale=True,
        n_x_bins=p_resolution,
        n_y_bins=pdot_resolution,
    )

    # Save density map file names into the partial dataset dictionary.
    ppdot_maps_dictionary.setdefault("input:" + map_name, []).append(
        ppdot_map_filename
    )

    log.info("{} generated...".format(ppdot_map_filename))
