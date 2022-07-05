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

from typing import List

import numpy as np

import pypopsyn.simulator.basics.constants as const

cfg = {}

# Function-specific profiling configuration.
cfg["enable_profiles"]: bool = False
cfg["show_profiles"]: bool = False
cfg["profiles_dir"]: str = "profiles"

# General profiling configuration.
cfg["profile_log"]: str = "profile.log"
cfg["profile_json"]: str = "profile.json"
cfg["show_profiling"]: bool = True

# Save time evolution output
cfg["save_dyn_evolution"]: bool = False
cfg["save_magrot_evolution"]: bool = False


# Initial population class parameters.

# Seed for the random number generation for the simulate_population_full.py.
cfg["seed_full"]: int = None
# Seed for the random number generation for the simulate_population_dyn.py.
cfg["seed_dyn"]: int = None
# Seed for the random number generation for the simulate_population_magrot_det.py.
cfg["seed_magrot"]: int = None

# Resolution for the spatial grid in the initial population.
cfg["resolution"]: int = 10000

# Integer number of neutron stars for the population.
cfg["NS_number"]: int = 300000

# Sun's distance from the galactocentric axis in [kpc].
cfg["R_sun"]: float = 8.3

# Sun's distance from the galactic plane in [kpc].
cfg["z_sun"]: float = 0.02

# Total radial extent of our simulation from the galactic center in [kpc].
cfg["r_extent"]: float = 20.0

# Total vertical extent of our simulation from the galactic plane in [kpc].
cfg["z_extent"]: float = 5.0

# Maximum kick velocity magnitude in [km/s].
cfg["vk_extent"]: float = 2500.0

# Number of spiral arms in the galaxy. If set to 5 the Local arm is included.
cfg["arm_number"]: int = 5

# Minimum age for the neutron stars in [yr].
cfg["t_age_min"]: float = 1.0

# Maximum age for the neutron stars in [yr].
cfg["t_age_max"]: float = 3e7

# Time step for the dynamical evolution [yr].
cfg["dyn_time_step"]: float = 1e4

# Galactic potential model used in the simulation. Choose between gmM19 or gmFK06.
cfg["galactic_model"]: str = "gmM19"

# Spiral arms model used in the simulation. Choose between saYMW17 or saFK06.
cfg["spiral_arms"]: str = "saYMW17"

# Model pdf for the radial density distribution of neutron star progenitors. Choose between "rmYK04" or "rmVV21".
cfg["radial_model"]: str = "rmYK04"

# Model pdf for the kick velocity. Choose between "km_maxwell", "km_exp", "km_2maxwell".
cfg["kick_model"]: str = "km_maxwell"

# Characteristic kick velocity in [km/s] for the exponential kick velocity pdf.
cfg["vk_c"]: float = 180.0

# Sigma in [km/s] for the Maxwell kick velocity pdf.
cfg["sigma_k"]: float = 265.0

# Characteristic height in [kpc] from the galactic plane.
cfg["h_c"]: float = 0.18


# Canonical neutron star parameters.

# Characteristic neutron star radius in [cm].
cfg["NS_radius"]: float = 1.1e6

# Characteristic neutron star mass in [g].
cfg["NS_mass"]: float = 1.4 * const.M_SUN


# Field, misalignment angle and period evolution parameters for a crust-based model.

# Mean and standard deviation for the Gaussian distributed initial periods in [s].
cfg["P_initial_mean"]: float = 0.3
cfg["P_initial_sigma"]: float = 0.2

# Mean and standard deviation for the log-normally distributed initial magnetic fields in [s].
cfg["B_initial_log10_mean"]: float = 13.25
cfg["B_initial_log10_sigma"]: float = 0.75

# Dimensionless coefficients k_0, k_1, k_2 for a force-free magnetosphere
# taken from Spitkovsky (2006) and Philippov et al. (2014).
# For comparison, in vacuum k_0 = 0 and k_1 = k_2 = 2/3.
cfg["k_coefficients"]: List[float] = [1.0, 1.0, 1.0]

# Dominant conductivity based on phonon or impurity scattering, in [1/s].
# For details see Cumming et al. (2004) or Gourgouliatos and Cumming (2014).
cfg["sigma"]: float = 1e24

# Characteristic length scale of the magnetic field in [cm].
cfg["L"]: float = 1e5

# Characteristic electron density in [g/cm^3].
cfg["n_e"]: float = 1e35

# Time step for the magneto-rotational evolution [yr].
cfg["magrot_time_step_log10"]: float = 1e-2

# Parameters of the analytical expression use to mimic the simulated magnetic field evolution curves.
# Power-law indices.
cfg["a1"]: float = 0.14
cfg["a2"]: float = 3.0
# Timescale parameters, normalization and power-law index.
cfg["tau1_norm"]: float = 9.0e16
cfg["tau1_a"]: float = 1.0
cfg["tau2_norm"]: float = 6.2e11
cfg["tau2_a"]: float = 0.4
# Time in [yr] when transitioning from the simulated curve to the simple power-law evolution.
cfg["t_trans"]: float = 1.0e6
# Late time power-law index.
cfg["a_late_t"]: float = 1.0

# Parameters for a log-normal distribution of the magnetic fields of the old millisecond pulsars.
cfg["B_millisec_mean"] = 8.5
cfg["B_millisec_sigma"] = 0.5

# Radio emission model parameters.

# Distance from the center of the star where the radio emission is supposed to be generated in [cm].
cfg["r_em"]: float = 3.0e7

# Mean and standard deviation for the log-normally distributed radio luminosity in [erg s^(-1) Hz^(-1)].
cfg["L_radio_log10_mean"]: float = 35.0
cfg["L_radio_log10_sigma"]: float = 0.8
cfg["epsilon_L"]: float = 0.5

# Free electron density model for the Galaxy, choose between "ne2001" and "ymw16".
cfg["ed_model"]: str = "ymw16"

# Number of Galactic isolated neutron stars detected by the considered surveys.
cfg["detected_real_PMPS"]: int = 961
cfg["detected_real_SMPS"]: int = 172


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
