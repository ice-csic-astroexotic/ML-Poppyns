"""
    density map generation routines.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import logging
import typing

import numpy as np

import mlpoppyns.generator.maps.maps2d_generators as mg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
map_generators = {
    "array": mg.generate_density_matrix,
    "array_kde": mg.generate_kde_density_matrix,
    "image": mg.generate_density_image,
    "image_kde": mg.generate_kde_density_image,
}
# Set the corresponding extensions for the types of position maps.
extensions = {
    "array": "npy",
    "array_kde": "npy",
    "image": "png",
    "image_kde": "png",
}

log = logging.getLogger(__name__)


def generate_density_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    x: np.array,
    y: np.array,
    x_resolution: int,
    y_resolution: int,
    x_log_scale: bool,
    y_log_scale: bool,
    maps_dictionary: dict,
    x_limits: typing.Tuple[float, float],
    y_limits: typing.Tuple[float, float],
    valid_simulation: bool,
) -> None:
    """
    This method generates a density map with great flexibility, the
    dimensions of the map can be chosen, whether using kernel density estimation (KDE) or not and
    the type (image or array) can also be decided, and the limits and resolution for it can be specified.
    As a result, a map with the specified filename and an extension determined by the chosen
    type is created as output.

    The specified dictionary of density maps for the dataset is also updated with the
    newly generated map.

    Args:
        dataset_path (str): Path to the folder where the map will be created.
        map_name (str): Specific name for this map.
        sample_number (int): Number to suffix this map in the dataset.
        map_type (str): Type of map to generate (array, array_kde, image or image_kde).
        x (np.array): Horizontal coordinate values for the points.
        y (np.array): Vertical coordinate values for the points.
        x_resolution (int): Resolution in the horizontal axis.
        y_resolution (int): Resolution in the vertical axis.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        maps_dictionary (dict): Dictionary of P-flux maps where to append the path of the new generated map.
        x_limits (Tuple[float, float]): Limits of the horizontal axis.
        y_limits (Tuple[float, float]): Limits of the vertical axis.
        valid_simulation (bool): If False the generated map will contain NaN values.
    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    map_generators[map_type](
        x,
        x_limits,
        y,
        y_limits,
        map_filename,
        x_log_scale,
        y_log_scale,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
        valid_simulation=valid_simulation,
    )

    # Save density map file path into the dataset dictionary.
    maps_dictionary.setdefault("input:" + map_name, []).append(map_filename)

    log.info("{} generated...".format(map_filename))
