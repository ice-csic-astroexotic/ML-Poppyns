"""
Simulator configuration

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""


cfg = {}

# Initial population class parameters.

cfg["R_sun"] = 8.3  # Sun distance from the galactocentric axis [kpc].
cfg["z_sun"] = 0.02  # Sun distance from the Galactic plane [kpc].
cfg["r_extent"] = 20.0  # Total radial extent from the galactic centre [kpc].
cfg["z_extent"] = 5.0  # Total vertical extent from the galactic plane [kpc].
cfg["vk_extent"] = 2500.0  # Maximum kick velocity magnitude [km/s].
cfg["seed"] = 42  # Seed for random number generation.
cfg["resolution"] = 10000  # Resolution for the grid in the initial population.
cfg[
    "NS_number"
] = 500000  # Integer number of neutron stars for the population.
cfg["arm_number"] = 4  # Number of spiral arms in the galaxy.
cfg["t_age_min"] = 1.0  # Minimum age for the neutron stars [yr].
cfg["t_age_max"] = 1e8  # Maximum age for the neutron stars [yr].
# Galactic potential model to use in the simulation. Choose between gmM19 or gmFK06.
cfg["galactic_model"] = "gmM19"
# Spiral arms model to use in the simulation. Choose between saYMW17 or saFK06.
cfg["spiral_arms"] = "saYMW17"
# Model pdf for the kick velocity. Choose between "km_maxwell" or "km_exp".
cfg["kick_model"] = "km_maxwell"
# Characteristic kick velocity in km/s for the kick velocity exponential pdf.
cfg["vk_c"] = 380.0
# sigma in km/s for the kick velocity Maxwell pdf.
cfg["sigma_k"] = 265.0
# Characteristic height in kpc from the galactic plane.
cfg["h_c"] = 0.18


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
