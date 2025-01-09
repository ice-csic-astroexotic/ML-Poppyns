"""
    Simulator configuration.

        Authors:

            Alberto Garcia Garcia (garciagarcia@ice.csic.es)
            Vanessa Graber (graber@ice.csic.es)
"""

import logging
import sys
from typing import List

import pypopsyn.simulator.basics.constants as const

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

# Specify here the absolute path to the directory where the repository is saved.
# If launching experiments on one of the PIC servers set cfg["server_run"] = True.
cfg["server_run"] = False
if cfg["server_run"]:
    cfg[
        "path_to_software"
    ] = "/data/magnesia/software/MAGNESIA_population_synthesis"
    cfg["path_to_output"] = "/data/magnesia/common"
else:
    # Change the following parameters to your local path, e.g., something like
    # /home/michele/Documents/MAGNESIA_population_synthesis. Otherwise, some notebooks might not work!
    cfg["path_to_software"] = ""
    cfg["path_to_output"] = ""

if cfg["path_to_software"] == "":
    log.warning(
        "path_to_software variable not set. Remember to set the right absolute path_to_software in the "
        "pypopsyn/simulator/config_simulator.py file."
    )

# Seed for the random number generation for simulate_population_full.py.
cfg["seed_full"]: int = None
# Seed for the random number generation for simulate_population_dyn.py.
cfg["seed_dyn"]: int = None
# Seed for the random number generation for simulate_population_magrot_det.py.
cfg["seed_magrot"]: int = None
# Seed for the random number generation for memory_efficient_sampling.py
cfg["seed_sampling"] = None

# Resolution for the parameter grid when performing random sampling for the neutron star properties from a pdf distribution.
# This is used to sample the initial position in Galactocentric coordinates, the kick velocity, the initial magnetic field,
# the inclination angle and the line of sight for the radio beam intercept.
cfg["resolution"]: int = 10000

# Total number of neutron stars to simulate.
# Used only when running simulate_population_full.py and simulate_population_dyn.py.
cfg["NS_number"]: int = 300000

# Minimum and maximum ages for the neutron stars in [yr].
cfg["t_age_min"]: float = 1.0
cfg["t_age_max"]: float = 3e7

# ===================== CANONICAL NEUTRON STAR PARAMETERS ========================

# Characteristic neutron star radius in [cm].
cfg["NS_radius"]: float = 1.1e6

# Characteristic neutron star mass in [g].
cfg["NS_mass"]: float = 1.4 * const.M_SUN

# ===================== DYNAMICAL PARAMETERS ========================

# Galactic potential model used in the simulation. Choose between gmM19 or gmFK06.
cfg["galactic_model"]: str = "gmM19"

# Spiral arms model used in the simulation. Choose between saYMW17 or saFK06.
cfg["spiral_arms"]: str = "saYMW17"

# Number of spiral arms in the Galaxy. If set to 5 the Local arm is included.
cfg["arm_number"]: int = 5

# Sun's distance from the galactocentric axis in [kpc].
cfg["R_sun"]: float = 8.3

# Sun's distance from the galactic plane in [kpc].
cfg["z_sun"]: float = 0.02

# Model pdf for the radial density distribution of neutron star progenitors. Choose between "rmYK04" or "rmVV21".
cfg["radial_model"]: str = "rmYK04"

# Total radial extent of the initial distribution of neutron star progenitors from the galactic center in [kpc].
cfg["r_extent"]: float = 20.0

# Characteristic height in [kpc] from the galactic plane for the exponential disk model.
cfg["h_c"]: float = 0.18

# Total vertical extent of the initial distribution of neutron star progenitors from the galactic plane in [kpc].
cfg["z_extent"]: float = 5.0

# Model pdf for the kick velocity. Choose between "km_maxwell", "km_exp", "km_2maxwell".
cfg["kick_model"]: str = "km_maxwell"

# Maximum kick velocity magnitude in [km/s].
cfg["vk_extent"]: float = 2500.0

# Characteristic kick velocity in [km/s] for the exponential kick velocity pdf (Faucher-Giguère amd Kaspi 2006).
cfg["vk_c"]: float = 180.0

# Sigma in [km/s] for the Maxwell kick velocity pdf (Hobbs et al. 2005).
cfg["sigma_k"]: float = 265.0

# Parameters for the double Maxwell kick velocity pdf (see Eq. (5) and Section 4.2 in Igoshev 2020).
# The weight denotes the importance of the first Maxwell component relative to the whole pdf.
# Its value has to be in the range between 0 and 1.
cfg["sigma_k_1"]: float = 55.0
cfg["sigma_k_2"]: float = 334.0
cfg["kick_weight"]: float = 0.19

# Time step for the dynamical evolution [yr].
cfg["dyn_time_step"]: float = 1e4

# ===================== MAGNETO-ROTATIONAL PARAMETERS FOR A CRUST-BASED MODEL ========================

# Model pdf for the initial spin period. Choose between "normal", "log-normal".
cfg["spin_period_model"]: str = "log-normal"

# Mean and standard deviation for the Gaussian distributed initial periods in [s].
cfg["P_initial_mean"]: float = 0.3
cfg["P_initial_sigma"]: float = 0.2

# Mean and standard deviation for the log-normal distributed initial periods in [s].
# (default values are taken from Pardo-Araujo et al. 2025).
cfg["P_initial_log10_mean"]: float = -0.67
cfg["P_initial_log10_sigma"]: float = 0.55

# Model pdf for the initial magnetic field. Choose between "log-normal", "double_log-normal", "smooth_tophat".
cfg["magnetic_field_model"]: str = "log-normal"

# Minimum and maximum initial magnetic field strength in [G] to simulate.
cfg["B_initial_log10_min"]: float = 10.0
cfg["B_initial_log10_max"]: float = 16.0

# Mean and standard deviation for the log-normally distributed initial magnetic fields in [G]
# (default values are taken from Pardo-Araujo et al. 2025).
cfg["B_initial_log10_mean"]: float = 13.09
cfg["B_initial_log10_sigma"]: float = 0.5

# Means and standard deviations and relative weight for the double log-normally distributed
# initial magnetic fields in [G]. The weight denotes the importance of the first log-normal component
# relative to whole pdf. Its value has to be in the range between 0 and 1.
cfg["B_initial_log10_mean1"]: float = 13.02
cfg["B_initial_log10_sigma1"]: float = 0.49
cfg["B_initial_log10_mean2"]: float = 14.5
cfg["B_initial_log10_sigma2"]: float = 0.5
cfg["B_initial_weight"]: float = 0.8

# Parameters for the smooth top-hat with Gaussian rise and decay for the initial magnetic fields in [G].
cfg["B_initial_log10_rise_mean"]: float = 13.02
cfg["B_initial_log10_rise_sigma"]: float = 0.49
cfg["B_initial_log10_decay_mean"]: float = 14.8
cfg["B_initial_log10_decay_sigma"]: float = 0.2
cfg["B_initial_log10_slope"]: float = -2.0

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

# We fit a functional equation of the magneto-thermal evolution curves (more information can be found in
# pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/README.md):
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

# Late time power-law index (default value is taken from Pardo-Araujo et al. 2025).
cfg["a_late"]: float = -0.88

# Parameters for a log-normal distribution of the magnetic fields of the old millisecond pulsars.
cfg["B_millisec_mean"] = 8.5
cfg["B_millisec_sigma"] = 0.5


# ===================== RADIO EMISSION-MODEL PARAMETERS ========================

# Model for the radio beam aperture. Choose between "standard_period_cone" and "power-law_period_cone".
cfg["radio_beam_model"] = "standard_period_cone"

# Distance in [cm] from the center of the star to where the radio emission is generated (Johnston et al. 2020).
# This parameter is required only if the cfg["radio_beam_model"] is set to "standard_period_cone".
cfg["r_em"]: float = 3.0e7

# Parameters for the "power-law_period_cone" model (Maciesiak et al. 2012).
cfg[
    "rho_b_0"
] = 2.5  # Half opening angle of the radio beam in [deg] corresponding to a spin period of 1 s.
cfg["a_beam"] = -0.5  # Power-law exponent.


# Relevant parameters for the log-normally distributed luminosity, L.
# We have implemented two different prescriptions for the luminosity in the module
# pypopsyn/simulator/multiband_emission/emission_radio.py based on the luminosity depending on different parameters.
# One prescription follows Faucher-Giguère & Kaspi (2006) (pdf_luminosity_radio_ppdot) and assumes that L depends on
# the period and period derivative.
# The second one assumes that L depends directly on the loss of rotational energy
# (pdf_luminosity_radio_edot). If the luminosity is given by pdf_luminosity_radio_ppdot, then the units of
# L_radio_log10_mean are [erg s^(3 * epsilon_L - 1)]. Otherwise, L_0 has units of [ergs/s]. The default values are
# taken from Pardo-Araujo et al. 2025.
cfg[
    "L_radio_log10_mean"
]: float = 26.17  # [erg s^(- 1)] if pdf_luminosity_radio_edot is used.
# cfg["L_radio_log10_mean"]: float = 35.5 [erg s^(3 * epsilon_L - 1) ] if pdf_luminosity_radio_ppdot is used.
cfg["L_radio_log10_sigma"]: float = 0.8
cfg["epsilon_L"]: float = 0.68
cfg["Erot_dot_0"]: float = 1e29

# Spectral index following a normal distribution as in Posselt et al. (2023). We set the standard deviation to 0 to
# efficiently produce a fixed spectral index.
cfg["mean_spectral_index"] = -1.8
cfg["std_spectral_index"] = 0

# Free electron density model for the Galaxy, choose between "ne2001" and "ymw16".
cfg["ed_model"]: str = "ymw16"

# Number of Galactic isolated neutron stars detected by the considered surveys.
# To obtain these estimates, we removed extragalactic sources and those in globular clusters.
# To exclude recycled objects that we cannot model with our current framework,
# we also use a cut-off in period of P > 0.01s and period derivative of Pdot > 10^-19s/s.
# The latter however only applies to those objects with measured Pdot values,
# i.e., the counts below also include those pulsars with P > 0.01s that have no Pdot measurement.
cfg["detected_real_PMPS"]: int = 1045
cfg["detected_real_SMPS"]: int = 218
cfg["detected_real_htru_low_mid"]: int = 1037
cfg["detected_real_htru_high"]: int = 20

# Numbers of objects associated with the three pulsar surveys as followed up with the TPA programme on MeerKAT.
# For details see Posselt et al. (2023). Note these numbers are used in the pypopsyn/generator/generate_observed_data.py
# script and differ from those given in the full ATNF Pulsar Catalogue.
cfg["detected_meerkat_PMPS"]: int = 640
cfg["detected_meerkat_SMPS"]: int = 170
cfg["detected_meerkat_HTRU"]: int = 668


def update_configuration(new_configuration: dict) -> None:
    """
    Update current configuration with custom one.

    Update each key in the current configuration dictionary with another custom
    configuration dictionary and output each updated key-value pair.

    Args:
        new_configuration (dict): dictionary with custom configuration.
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
