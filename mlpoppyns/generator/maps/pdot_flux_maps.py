"""
    Pdot-flux map generation routines.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import logging
import typing

import numpy as np
import pypopsyn.generator.maps.maps2d_generator as mg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
pdot_fluxes_map_generators = {
    "array": mg.generate_density_matrix,
    "array_kde": mg.generate_density_matrix,
    "image": mg.generate_density_map,
    "image_kde": mg.generate_density_map,
}
# Set the corresponding extensions for the types of position maps.
extensions = {
    "array": "npy",
    "array_kde": "npy",
    "image": "png",
    "image_kde": "png",
}

log = logging.getLogger(__name__)


def generate_pdot_flux_map(
    dataset_path: str,
    map_name: str,
    sample_number: int,
    map_type: str,
    period_derivatives: np.array,
    fluxes: np.array,
    pdot_resolution: int,
    flux_resolution: int,
    pdot_flux_maps_dictionary: dict,
    pdot_limits: typing.Tuple[float, float] = (1.0e-20, 1.0e-9),
    flux_limits: typing.Tuple[float, float] = (1.0e-15, 1.0e-9),
) -> None:
    """
    This method generates a discrete Pdot-flux map with great flexibility, the
    dimensions of the map can be chosen, the type (image or array) can also be
    decided, and the limits and resolution for it can be specified. As a result,
    a map with the specified filename and an extension determined by the chosen
    type is created as output.

    The dictionary of P-flux maps for the dataset is also updated with the
    generated example.

    Args:
        dataset_path (str): Path to the folder where the map will be created.
        map_name (str): Specific name for this map.
        sample_number (int): Number to suffix this map in the dataset.
        map_type (str): Type of map to generate (array or image).
        period_derivatives (np.array): Spin period derivatives of the neutron stars (horizontal axis).
        fluxes (np.array): Fluxes of the neutron stars (vertical axis).
        pdot_resolution (int): Resolution in the horizontal axis.
        flux_resolution (int): Resolution in the vertical axis.
        pdot_flux_maps_dictionary (dict): Partial dictionary of P-Pdot maps.
        pdot_limits (Tuple[float, float]): Limits of the horizontal axis.
        flux_limits (Tuple[float, float]): Limits of the vertical axis.
    """

    # Compose the final filename with the dataset path, the name for the map,
    # the current sample suffix and the appropriate extension.
    pdot_flux_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    # Automagically select and call the appropriate generator depending on the
    # specified type for the map.
    pdot_fluxes_map_generators[map_type](
        period_derivatives,
        pdot_limits,
        fluxes,
        flux_limits,
        pdot_flux_map_filename,
        x_log_scale=True,
        y_log_scale=True,
        n_x_bins=pdot_resolution,
        n_y_bins=flux_resolution,
    )

    # Save density map file names into the partial dataset dictionary.
    pdot_flux_maps_dictionary.setdefault("input:" + map_name, []).append(
        pdot_flux_map_filename
    )

    log.info("{} generated...".format(pdot_flux_map_filename))
