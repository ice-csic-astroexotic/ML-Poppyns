"""
Simulator configuration.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


cfg = {}

# Function-specific profiling configuration.
cfg["enable_profiles"] = False
cfg["show_profiles"] = False
cfg["profiles_dir"] = "profiles"

# General profiling configuration.
cfg["profile_log"] = "profile.log"
cfg["profile_json"] = "profile.json"
cfg["show_profiling"] = True

# Initial population class parameters.

# Sun's distance from the galactocentric axis in [kpc].
cfg["R_sun"] = 8.3

# Sun's distance from the galactic plane in [kpc].
cfg["z_sun"] = 0.02

# Total radial extent of our simulation from the galactic center in [kpc].
cfg["r_extent"] = 20.0

# Total vertical extent of our simulation from the galactic plane in [kpc].
cfg["z_extent"] = 5.0

# Maximum kick velocity magnitude in [km/s].
cfg["vk_extent"] = 2500.0

# Seed for the random number generation.
cfg["seed"] = None

# Resolution for the spatial grid in the initial population.
cfg["resolution"] = 10000

# Integer number of neutron stars for the population.
cfg["NS_number"] = 100000

# Number of spiral arms in the galaxy.
cfg["arm_number"] = 4

# Minimum age for the neutron stars in [yr].
cfg["t_age_min"] = 1.0

# Maximum age for the neutron stars in [yr].
cfg["t_age_max"] = 1e7

# Galactic potential model used in the simulation. Choose between gmM19 or gmFK06.
cfg["galactic_model"] = "gmM19"

# Spiral arms model used in the simulation. Choose between saYMW17 or saFK06.
cfg["spiral_arms"] = "saYMW17"

# Model pdf for the kick velocity. Choose between "km_maxwell" or "km_exp".
cfg["kick_model"] = "km_maxwell"

# Characteristic kick velocity in [km/s] for the exponential kick velocity pdf.
cfg["vk_c"] = 380.0

# Sigma in [km/s] for the Maxwell kick velocity pdf.
cfg["sigma_k"] = 265.0

# Characteristic height in [kpc] from the galactic plane.
cfg["h_c"] = 0.18


def update_configuration(new_configuration) -> None:
    """
    Update current configuration with custom one.

    Update each key in the current configuration dictionary with another custom
    configuration dictionary and output each updated key-value pair.

    Args:

        new_configuration: dictionary with custom configuration.

    Returns:

        Nothing.

    """

    for key, value in new_configuration.items():

        if key not in cfg.keys():

            raise ValueError(
                "Trying to update non-existing configuration key {}".format(
                    key
                )
            )

        print(
            "Updating key {} in configuration with value {}...".format(
                key, value
            )
        )

        cfg[key] = value
