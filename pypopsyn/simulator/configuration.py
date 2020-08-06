"""
Simulator configuration.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)

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

import numpy as np

import pypopsyn.simulator.basics.constants as const

cfg = {}

# Function-specific profiling configuration.
cfg["enable_profiles"] = False
cfg["show_profiles"] = False
cfg["profiles_dir"] = "profiles"

# General profiling configuration.
cfg["profile_log"] = "profile.txt"
cfg["show_profiling"] = False


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


# Canonical neutron star parameters.

# Characteristic neutron star radius in [cm].
cfg["NS_radius"] = 1.1e6

# Characteristic neutron star mass in solar masses.
cfg["NS_mass"] = 1.4 * const.M_SUN


# Field, misalignment angle and period evolution parameters for a crust-based model.

# Mean and standard deviation for the Gaussian distributed initial periods in [s].
cfg["P_initial_mean"] = 0.22
cfg["P_initial_sigma"] = 0.42

# Mean and standard deviation for the log-normally distributed initial magnetic fields in [s].
cfg["B_initial_log10_mean"] = 13.20
cfg["B_initial_log10_sigma"] = 0.62

# Dimensionless coefficients k_0, k_1, k_2 for a force-free magnetosphere
# taken from Spitkovsky (2006) and Philippov et al. (2014).
# For comparison, in vacuum k_0 = 0 and k_1 = k_2 = 2/3.
cfg["k_coefficients"] = np.array([1.0, 1.0, 1.0])

# Dominant conductivity based on phonon or impurity scattering, in [1/s].
# For details see Cumming et al. (2004) or Gourgouliatos and Cumming (2014).
cfg["sigma"] = 1e24

# Characteristic length scale of the magnetic field in [cm].
cfg["L"] = 1e5

# Characteristic electron density in [g/cm^3].
cfg["n_e"] = 1e36

# Time step for the magneto-rotational evolution [yr].
cfg["time_step"] = 1e3


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
