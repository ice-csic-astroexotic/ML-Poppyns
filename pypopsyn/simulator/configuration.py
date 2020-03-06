""" Simulator configuration.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import pathlib

cfg = {}

# number of samples to create
cfg["n_samples"] = 1

# initial population class parameters
cfg["r_extent"] = 20.0  # Total radial extent from the galactic centre [kpc].
cfg["z_extent"] = 5.0  # Total vertical extent from the galactic plane [kpc].
cfg["vp_extent"] = 2000.0  # Maximum proper velocity magnitude [km/s].
cfg["resolution"] = 10000  # Resolution for the grid in the initial population.
cfg["NS_number"] = 50000  # Number of neutron stars for the population.
cfg["arm_number"] = 4  # Number of spiral arms in the galaxy.
cfg["t_age_min"] = 1.0  # Minimum age for the neutron stars [yr].
cfg["t_age_max"] = 1.0e8  # Maximum age for the neutron stars [yr].

# initial velocity parameters
cfg["vp_mean"] = 380.0  # characteristic kick velocity in km/s for the proper
# velocity pdf.

# path where the generated population are stored
cfg["pop_path"] = pathlib.Path("multiran")


def update_configuration(new_configuration) -> None:

    """ Update current configuration with custom one.

    Update each key in the current configuration dictionary with another custom
    configuration dictionary and output each updated key-value pair.

    Args:

        new_configuration: dictionary with custom configuration.

    Returns:

        Nothing.

    """

    for key, value in new_configuration.items():

        print(
            "Updating key {} in configuration with value {}...".format(
                key, value
            )
        )

        cfg[key] = value
