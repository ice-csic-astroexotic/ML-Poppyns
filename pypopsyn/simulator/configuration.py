""" Simulator configuration.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

cfg = {}

cfg["r_extent"] = 20.0  # [kpc]
cfg["z_extent"] = 5.0  # [kpc]
cfg["vp_extent"] = 2000.0  # [kpc]
cfg["resolution"] = 10000  # TODO: needs units
cfg["NS_number"] = 50000
cfg["arm_number"] = 4


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
