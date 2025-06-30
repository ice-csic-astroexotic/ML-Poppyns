"""
    Velocity maps generation routines.

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
velocity_map_generators = {
    "array": mg.generate_avg_weight_matrix,
    "image": mg.generate_avg_weight_map,
}
# Set the corresponding extensions for the types of velocity maps.
extensions = {"array": "npy", "image": "png"}

log = logging.getLogger(__name__)


def generate_velocity_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    x_positions: np.array,
    y_positions: np.array,
    velocities: np.array,
    velocity_avg_min: float,
    x_resolution: int,
    y_resolution: int,
    velocity_maps_dictionary: dict,
    x_limits: typing.Tuple[float, float] = (-20.0, 20.0),
    y_limits: typing.Tuple[float, float] = (-20.0, 20.0),
) -> None:
    """
    This method generates a discrete velocity map with great flexibility, the
    dimensions of the map can be chosen, the type (image or array) can also be
    decided, and the limits and resolution for it can be specified. As a result,
    a map with the specified filename and an extension determined by the chosen
    type is created as output.

    The dictionary of velocity maps for the dataset is also updated with the
    generated example.

    Args:
        dataset_path (str): Path to the folder where the map will be created.
        map_name (str): Specific name for this map.
        sample_number (int): Number to suffix this map in the dataset.
        map_type (str): Type of map to generate (array or image).
        x_positions (np.array): Positions in the first axis (horizontal).
        y_positions (np.array): Positions in the second axis (vertical).
        velocities (np.array): Array of velocities to put in the map.
        velocity_avg_min (float): Minimum average value of the velocities to put in the map in the empty bins.
        x_resolution (int): Resolution in the horizontal axis.
        y_resolution (int): Resolution in the vertical axis.
        velocity_maps_dictionary (dict): Dictionary of velocity maps.
        x_limits (Tuple[float, float]): Limits of the horizontal axis.
        y_limits (Tuple[float, float]): Limits of the vertical axis.
    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    velocity_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    velocity_map_generators[map_type](
        x_positions,
        x_limits,
        y_positions,
        y_limits,
        velocities,
        velocity_avg_min,
        velocity_map_filename,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
    )

    # Save velocity map file names into the partial dataset dictionary.
    velocity_maps_dictionary.setdefault("input:" + map_name, []).append(
        velocity_map_filename
    )

    log.info("{} generated...".format(velocity_map_filename))
