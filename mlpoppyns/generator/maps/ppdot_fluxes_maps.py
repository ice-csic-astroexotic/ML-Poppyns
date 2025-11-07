"""
    Average flux ppdot map generation routines.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import logging
import typing

import numpy as np

import mlpoppyns.generator.maps.maps2d_generator as mg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
ppdot_fluxes_map_generators = {
    "array": mg.generate_avg_weight_matrix,
    "array_kde": mg.generate_kde_weight_matrix,
    "image": mg.generate_avg_weight_map,
    "image_kde": mg.generate_kde_weight_map,
}
# Set the corresponding extensions for the types of position maps.
extensions = {
    "array": "npy",
    "array_kde": "npy",
    "image": "png",
    "image_kde": "png",
}

log = logging.getLogger(__name__)


def generate_ppdot_fluxes_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    x_positions: np.array,
    y_positions: np.array,
    fluxes: np.array,
    flux_avg_min: float,
    x_resolution: int,
    y_resolution: int,
    ppdot_fluxes_maps_dictionary: dict,
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
        x_positions (np.array): Positions in the first axis (horizontal).
        y_positions (np.array): Positions in the second axis (vertical).
        fluxes (np.array): Array of the logarithm of the fluxes to put in the map.
        flux_avg_min (float): Minimum average value of the logarithm of the fluxes to put in the map in the empty bins.
        x_resolution (int): Resolution in the horizontal axis.
        y_resolution (int): Resolution in the vertical axis.
        ppdot_fluxes_maps_dictionary (dict): Dictionary of average flux P-Pdot maps where to append the path of the new generated map.
        x_limits (Tuple[float, float]): Limits of the horizontal axis.
        y_limits (Tuple[float, float]): Limits of the vertical axis.
    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    ppdot_fluxes_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    ppdot_fluxes_map_generators[map_type](
        x_positions,
        x_limits,
        y_positions,
        y_limits,
        fluxes,
        flux_avg_min,
        ppdot_fluxes_map_filename,
        x_log_scale=True,
        y_log_scale=True,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
    )

    # Save file names of ppdot-flux maps into the partial dataset dictionary.
    ppdot_fluxes_maps_dictionary.setdefault("input:" + map_name, []).append(
        ppdot_fluxes_map_filename
    )

    log.info("{} generated...".format(ppdot_fluxes_map_filename))
