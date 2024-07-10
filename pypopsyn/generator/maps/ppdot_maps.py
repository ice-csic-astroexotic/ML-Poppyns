"""
    P-Pdot maps generation routines.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
"""

import logging
import typing

import numpy as np

import pypopsyn.generator.maps.maps2d_generator as mg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
ppdot_map_generators = {
    "array": mg.generate_density_matrix,
    "image": mg.generate_density_map,
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
) -> None:
    """
    This method generates a discrete P-Pdot diagram map with great flexibility, the
    dimensions of the map can be chosen, the type (image or array) can also be
    decided, and the limits and resolution for it can be specified. As a result,
    a map with the specified filename and an extension determined by the chosen
    type is created as output.

    The dictionary of P-Pdot maps for the dataset is also updated with the
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
        ppdot_maps_dictionary (dict): partial dictionary of P-Pdot maps.
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
