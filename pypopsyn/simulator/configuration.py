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

import pypopsyn.simulator.basics.constants as const

cfg = {}


# ===================== GENERAL SIMULATION PARAMETERS ========================

# Function-specific profiling configuration.
cfg["enable_profiles"]: bool = False
cfg["show_profiles"]: bool = False
cfg["profiles_dir"]: str = "profiles"

# General profiling configuration.
cfg["profile_log"]: str = "profile.log"
cfg["profile_json"]: str = "profile.json"
cfg["show_profiling"]: bool = True

# Save time evolution output.
cfg["save_dyn_evolution"]: bool = False
cfg["save_magrot_evolution"]: bool = False

# To run the simulations in the server with HTCondor set cfg["server_run"] = True.
cfg["server_run"] = False

if cfg["server_run"]:
    cfg[
        "path_server_software"
    ] = "/data/magnesia/software/MAGNESIA_population_synthesis/"
    cfg["path_server_output"] = "/data/magnesia/common/"
else:
    cfg["path_server_software"] = ""
    cfg["path_server_output"] = ""

# ===================== INITIAL POPULATION CLASS PARAMETERS ========================

# Seed for the random number generation for simulate_population_full.py.
cfg["seed_full"]: int = None
# Seed for the random number generation for simulate_population_dyn.py.
cfg["seed_dyn"]: int = None
# Seed for the random number generation for simulate_population_magrot_det.py.
cfg["seed_magrot"]: int = None
# Seed for the random number generation for memory_efficient_sampling.py
cfg["seed_sampling"] = None

# Resolution for the spatial grid in the initial population.
cfg["resolution"]: int = 10000

# Integer number of neutron stars for the population.
cfg["NS_number"]: int = 300000

# ODE solver tolerance for Julia.
cfg["ODE_solver_tol"] = 1e-8

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

# Number of spiral arms in the Galaxy. If set to 5 the Local arm is included.
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


# ===================== CANONICAL NEUTRON STAR PARAMETERS ========================

# Characteristic neutron star radius in [cm].
cfg["NS_radius"]: float = 1.1e6

# Characteristic neutron star mass in [g].
cfg["NS_mass"]: float = 1.4 * const.M_SUN


# ===================== MAGNETO-ROTATIONAL PARAMETERS FOR A CRUST-BASED MODEL ========================

# Model pdf for the initial spin period. Choose between "normal", "log-normal".
cfg["spin_period_model"]: str = "log-normal"

# Mean and standard deviation for the Gaussian distributed initial periods in [s].
cfg["P_initial_mean"]: float = 0.3
cfg["P_initial_sigma"]: float = 0.2

# Mean and standard deviation for the log-normal distributed initial periods in [s].
cfg["P_initial_log10_mean"]: float = -0.6
cfg["P_initial_log10_sigma"]: float = 0.3

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


# ===================== FIT PARAMETERS FOR MAGNETO-THERMAL SIMULATIONS ========================

# For the magneto-thermal simulations the following set-up was employed:
# The equation of state is SLy4 with a NS mass of 1.4 Msun and radius of 11.74 km.
# The impurity parameter in the pasta layer is fixed to 100. For the impurity in the outer and inner crust
# (excluding the pasta layer), the fits of Carreau et al. (2020) have been used (see Figure 5 in that paper).
# The envelope model is taken from Potekhin et al. (2015).
# Superfluid and superconducting gap parametrizations are taken from Ho et al. (2015):
# SFB for crustal neutrons, TToa for core neutrons and CCDKp for core protons.
# We fit a functional equation of the form:
# B(t) = B_initial * (1 + t/tau1)**a1 * (1 + t/tau2)**(a2-a1) * (1 + t/tau_late)**(a_late-a2)
# with tau1 = A1 * B_initial**b1 and tau2 = A2 * B_initial**b2.

# Power-law indices.
cfg["a1"]: float = -0.13
cfg["a2"]: float = -3.0

# Timescale parameters, normalizations and power-law indices.
cfg["A1"]: float = 1.0e14
cfg["b1"]: float = -0.8
cfg["A2"]: float = 6.0e8
cfg["b2"]: float = -0.2

# Timescale in [yr] when transitioning from the simulated curves to the simple late-time power-law evolution.
cfg["tau_late"]: float = 2.0e6

# Late time power-law index.
cfg["a_late"]: float = -2.0

# Parameters for a log-normal distribution of the magnetic fields of the old millisecond pulsars.
cfg["B_millisec_mean"] = 8.5
cfg["B_millisec_sigma"] = 0.5


# ===================== RADIO EMISSION-MODEL PARAMETERS ========================

# Distance from the center of the star where the radio emission is supposed to be generated in [cm]
# (see Johnston et al. 2020).
cfg["r_em"]: float = 3.0e7

# Mean and standard deviation for the log-normally distributed radio luminosity normalization factor
# Given in [erg s^(3 * epsilon_L - 1) ]. Parameters were adjusted to match observed data.
cfg["L_radio_log10_mean"]: float = 35.5
cfg["L_radio_log10_sigma"]: float = 0.8
cfg["epsilon_L"]: float = 0.5

# Free electron density model for the Galaxy, choose between "ne2001" and "ymw16".
cfg["ed_model"]: str = "ymw16"

# Number of Galactic isolated neutron stars detected by the considered surveys.
cfg["detected_real_PMPS"]: int = 1009
cfg["detected_real_SMPS"]: int = 218
cfg["detected_real_htru_low_mid"]: int = 1023
cfg["detected_real_htru_high"]: int = 20


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
