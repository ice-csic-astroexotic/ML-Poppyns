import logging

import pypopsyn.generator.dataset_generator as dg

# Initialize the multiple options we have to generate the different data
# inputs which will be later selected at runtime depending on the arguments.
velocity_map_generators = {
    "array": dg.generate_avg_weight_matrix,
    "image": dg.generate_avg_weight_map,
}
extensions = {"array": "npy", "image": "png"}

log = logging.getLogger(__name__)


def generate_velocity_map(
    dataset_path,
    map_name,
    sample_number,
    map_type,
    x_positions,
    y_positions,
    velocities,
    x_resolution,
    y_resolution,
    normalize,
    velocity_maps_dictionary,
    x_limits=(-20.0, 20.0),
    y_limits=(-20.0, 20.0),
):

    velocity_map_filename = "{}/{}_{}.{}".format(
        dataset_path, map_name, sample_number, extensions[map_type]
    )

    velocity_map_generators[map_type](
        x_positions,
        x_limits,
        y_positions,
        y_limits,
        velocities,
        velocity_map_filename,
        n_x_bins=x_resolution,
        n_y_bins=y_resolution,
        normalize=normalize,
    )

    # save velocity map filenames into a dictionary
    velocity_maps_dictionary.setdefault("input:" + map_name, []).append(
        velocity_map_filename
    )

    log.info("{} generated...".format(velocity_map_filename))
