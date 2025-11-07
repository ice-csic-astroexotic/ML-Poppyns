"""
    flux maps generation routines.

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
    "array": mg.generate_avg_weight_matrix,
    "array_kde": mg.generate_kde_weight_matrix,
    "image": mg.generate_avg_weight_image,
    "image_kde": mg.generate_kde_weight_image,
}
# Set the corresponding extensions for the types of position maps.
extensions = {
    "array": "npy",
    "array_kde": "npy",
    "image": "png",
    "image_kde": "png",
}

log = logging.getLogger(__name__)


def generate_weighted_density_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    x: np.array,
    y: np.array,
    w: np.array,
    w_min: float,
    x_resolution: int,
    y_resolution: int,
    x_log_scale: bool,
    y_log_scale: bool,
    maps_dictionary: dict,
    x_limits: typing.Tuple[float, float] = (1e-2, 1e2),
    y_limits: typing.Tuple[float, float] = (1e-20, 1e-9),
) -> None:
    """
    This method generates a flux-averaged ppdot map. The dimensions of the map
    can be chosen, the type (image or array) can also be decided, and the limits
    and resolution for it can be specified. As a result, a map with the specified
    filename and an extension determined by the chosen type is created as output.

    The dictionary of flux ppdot maps for the dataset is also updated with the
    generated example.

    Args:
        dataset_path (str): Path to the folder where the map will be created.
        map_name (str): Specific name for this map.
        sample_number (int): Number to suffix this map in the dataset.
        map_type (str): Type of map to generate (array or image).
        x (np.array): Horizontal coordinate values for the points.
        y (np.array): Vertical coordinate values for the points.
        w (np.array): Array of the logarithm of the fluxes to put in the map.
        w_min (float): Either the minimum average value to assign to the empty bins or, for KDE maps, the minimum weight
            value to subtract to ensure positivity of the weights passed to the KDE.
        x_resolution (int): Resolution in the horizontal axis.
        y_resolution (int): Resolution in the vertical axis.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        maps_dictionary (dict): Dictionary of the maps where to append the path of the new generated map.
        x_limits (Tuple[float, float]): Limits of the horizontal axis.
        y_limits (Tuple[float, float]): Limits of the vertical axis.
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
        w,
        w_min,
        map_filename,
        x_log_scale,
        y_log_scale,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
    )

    # Save file names of ppdot-flux maps into the partial dataset dictionary.
    maps_dictionary.setdefault("input:" + map_name, []).append(map_filename)

    log.info("{} generated...".format(map_filename))
