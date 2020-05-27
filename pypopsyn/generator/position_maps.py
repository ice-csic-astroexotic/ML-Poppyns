import logging

import pypopsyn.generator.dataset_generator as dg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
position_map_generators = {
    "array": dg.generate_density_matrix,
    "image": dg.generate_density_map,
}
extensions = {"array": "npy", "image": "png"}

log = logging.getLogger(__name__)


def generate_position_map(
    dataset_path,
    map_name,
    sample_number,
    map_type,
    x_positions,
    y_positions,
    x_resolution,
    y_resolution,
    normalize,
    position_maps_dictionary,
    x_limits=(-20.0, 20.0),
    y_limits=(-20.0, 20.0),
):

    position_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    position_map_generators[map_type](
        x_positions,
        x_limits,
        y_positions,
        y_limits,
        position_map_filename,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
        normalize=normalize,
    )

    # Save density map filenames into a dictionary.
    position_maps_dictionary.setdefault(
        "input:" + position_map_filename, []
    ).append(position_map_filename)

    log.info("{} generated...".format(position_map_filename))
