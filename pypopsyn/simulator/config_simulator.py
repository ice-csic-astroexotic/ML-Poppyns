"""
    Simulator configuration.

        Authors:

            Alberto Garcia Garcia (garciagarcia@ice.csic.es)
            Vanessa Graber (graber@ice.csic.es)
"""

import logging
import pathlib
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
cfg["server_run"]: bool = False
if cfg["server_run"]:
    cfg[
        "path_to_software"
    ]: str = "/data/magnesia/software/MAGNESIA_population_synthesis"
    cfg["path_to_output"]: str = "/data/magnesia/common"
else:
    # Change the following parameters to your local path, e.g., something like
    # /home/michele/Documents/MAGNESIA_population_synthesis. Otherwise, some notebooks might not work!
    cfg["path_to_software"]: str = ""
    cfg["path_to_output"]: str = ""

if cfg["path_to_software"] == "":
    log.warning(
        "path_to_software variable not set. Remember to set the right absolute path_to_software in the file: "
        "pypopsyn/simulator/config_simulator.py."
    )

# Seed for the random number generation for simulate_population_full.py.
cfg["seed_full"]: int = None
# Seed for the random number generation for simulate_population_dyn.py.
cfg["seed_dyn"]: int = None
# Seed for the random number generation for simulate_population_magrot_det.py.
cfg["seed_magrot"]: int = None
# Seed for the random number generation for memory_efficient_sampling.py
cfg["seed_sampling"]: bool = None

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

# Maximum birth rate in neutron stars per century that can be reached by a simulation before stopping.
cfg["birth_rate_max"]: float = 5.0

# Flag indicating whether to perform the X-ray simulation or not. If set to True then radio and X-ray emissions
# will be simulated; if False then only the radio population synthesis will be performed.
cfg["simulation_xray"]: bool = True

# ===================== CANONICAL NEUTRON STAR PARAMETERS ========================

# Characteristic neutron star mass in [g].
cfg["NS_mass"]: float = 1.4 * const.M_SUN

# ===================== DYNAMICAL PARAMETERS ========================

# Galactic potential model used in the simulation. Choose between gmM19 or gmFK06.
cfg["galactic_model"]: str = "gmM19"

# Flag indicating whether to use the electron density model of Yao et al. (2017) or not to sample the initial positions
# of neutron stars. If set to False the spiral-arm implementation right below will be used.
cfg["sample_edm"]: bool = True

if not cfg["sample_edm"]:
    # Spiral arms model used in the simulation. Choose between saYMW17 or saFK06.
    cfg["spiral_arms"]: str = "saYMW17"

    # Number of spiral arms in the Galaxy. If set to 5 the Local arm is included.
    cfg["arm_number"]: int = 5

    # Model pdf for the radial density distribution of neutron star progenitors. Choose between "rmYK04" or "rmVV21".
    cfg["radial_model"]: str = "rmYK04"

# Sun's distance from the galactocentric axis in [kpc].
cfg["R_sun"]: float = 8.3

# Sun's distance from the galactic plane in [kpc].
cfg["z_sun"]: float = 0.02

# Total radial extent of the initial distribution of neutron star progenitors from the galactic center in [kpc].
cfg["r_extent"]: float = 20.0

# Characteristic height in [kpc] from the galactic plane for the exponential disk model.
cfg["h_c"]: float = 0.18

# Total vertical extent of the initial distribution of neutron star progenitors from the galactic plane in [kpc].
cfg["z_extent"]: float = 5.0

# Model pdf for the kick velocity. Choose between "km_maxwell", "km_exp", "km_double_maxwell".
cfg["kick_model"]: str = "km_maxwell"

# Maximum kick velocity magnitude in [km/s].
cfg["vk_extent"]: float = 2500.0

if cfg["kick_model"] == "km_exp":
    # Characteristic kick velocity in [km/s] for the exponential kick velocity pdf (Faucher-Giguère amd Kaspi 2006).
    cfg["vk_c"]: float = 180.0

elif cfg["kick_model"] == "km_maxwell":
    # Sigma in [km/s] for the Maxwell kick velocity pdf (Hobbs et al. 2005).
    cfg["sigma_k"]: float = 265.0

elif cfg["kick_model"] == "km_double_maxwell":
    # Parameters for the double Maxwell kick velocity pdf (see Eq. (5) and Section 4.2 in Igoshev 2020).
    # The weight denotes the importance of the first Maxwell component relative to the whole pdf.
    # Its value has to be in the range between 0 and 1.
    cfg["sigma_k_comp1"]: float = 55.0
    cfg["sigma_k_comp2"]: float = 334.0
    cfg["kick_weight_comp1"]: float = 0.19

else:
    log.error(
        "The specified model for the kick velocity distribution is not supported."
        "Please choose between km_maxwell, km_exp or km_double_maxwell."
    )

# Time step for the dynamical evolution [yr].
cfg["dyn_time_step"]: float = 1e4

# ===================== MAGNETO-ROTATIONAL PARAMETERS FOR A CRUST-BASED MODEL ========================

# Model pdf for the initial spin period. Choose between "normal", "log-normal".
cfg["spin_period_model"]: str = "log-normal"

if cfg["spin_period_model"] == "normal":
    # Mean and standard deviation for the Gaussian distributed initial periods in [s].
    cfg["P_initial_mean"]: float = 0.3
    cfg["P_initial_sigma"]: float = 0.2

elif cfg["spin_period_model"] == "log-normal":
    # Mean and standard deviation for the log-normal distributed initial periods in [s].
    # (default values are taken from Pardo-Araujo et al. 2025).
    cfg["P_initial_log10_mean"]: float = -0.67
    cfg["P_initial_log10_sigma"]: float = 0.55

else:
    log.error(
        "The specified model for the initial spin period distribution is not supported."
        "Please choose between normal or log-normal."
    )

# Model pdf for the initial magnetic field. Choose between "log-normal", "double_log-normal", "smooth_tophat".
cfg["magnetic_field_model"]: str = "log-normal"

# Minimum and maximum initial magnetic field strength in [G] to simulate.
cfg["B_initial_log10_min"]: float = 10.0
cfg["B_initial_log10_max"]: float = 16.0

if cfg["magnetic_field_model"] == "log-normal":
    # Mean and standard deviation for the log-normally distributed initial magnetic fields in [G]
    # (default values are taken from Pardo-Araujo et al. 2025).
    cfg["B_initial_log10_mean"]: float = 13.09
    cfg["B_initial_log10_sigma"]: float = 0.5

elif cfg["magnetic_field_model"] == "double_log-normal":
    # Means and standard deviations and relative weight for the double log-normally distributed
    # initial magnetic fields in [G]. The weight denotes the importance of the first log-normal component
    # relative to whole pdf. Its value has to be in the range 0 and 1.
    cfg["B_initial_log10_mean_comp1"]: float = 13.09
    cfg["B_initial_log10_sigma_comp1"]: float = 0.5
    cfg["B_initial_log10_mean_comp2"]: float = 14.3
    cfg["B_initial_log10_sigma_comp2"]: float = 0.3
    cfg["B_initial_log10_weight_comp1"]: float = 0.5

elif cfg["magnetic_field_model"] == "smooth_tophat":
    # Parameters for the smooth top-hat with Gaussian rise and decay for the initial magnetic fields in [G].
    cfg["B_initial_log10_rise_mean"]: float = 13.02
    cfg["B_initial_log10_rise_sigma"]: float = 0.49
    cfg["B_initial_log10_decay_mean"]: float = 14.5
    cfg["B_initial_log10_decay_sigma"]: float = 0.2
    cfg["B_initial_log10_slope"]: float = 0.0

else:
    log.error(
        "The specified model for the initial magnetic-field distribution is not supported."
        "Please choose between log-normal, double_log-normal, or smooth_tophat."
    )

# Dimensionless coefficients k_0, k_1, k_2 for a force-free magnetosphere
# taken from Spitkovsky (2006) and Philippov et al. (2014).
# For comparison, in vacuum k_0 = 0 and k_1 = k_2 = 2/3.
cfg["k_coefficients"]: List[float] = [1.0, 1.0, 1.0]

# Time step for the magneto-rotational evolution [yr].
cfg["magrot_time_step_log10"]: float = 1e-2

# ===================== FIT PARAMETERS FOR MAGNETO-THERMAL SIMULATIONS ========================

# Model for the magneto-thermal simulations. Choose between "analytical" (in which case the magnetic field evolution
# is determined analytically from the evolution of a generalised induction equation), or fits to numerical simulations,
# specifically "SLy4_dip-tor_heavy", "BSk24_dip-tor_heavy", "BSk24_dip-tor_light", "BSk24_multi_heavy" and
# "BSk24_multi_light".
cfg["magneto-thermal_model"]: str = "BSk24_dip-tor_heavy"

# If the model "analytical" is chosen we use the approximated solution of Aguilera et al. (2008) for the evolution of
# the magnetic field.

# In the case of all other models, we fit a functional equation of the magnetic field evolution curves
# (more information can be found in pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/README.md):
# B(t) = B_initial * (1 + t/tau1)**a1 * (1 + t/tau2)**(a2-a1) * (1 + t/tau_late)**(a_late-a2)
# with tau1 = A1 * B_initial**b1 and tau2 = A2 * B_initial**b2.
# The fit parameters for each model were adjusted by hand (see the notebook
# tutorials/analysis_notebooks/magnetic_field_evolution_fit.ipynb for more details).

if cfg["magneto-thermal_model"] == "analytical":
    # Set the characteristic neutron star radius in [cm] for a mass of 1.4 Msun.
    cfg["NS_radius"]: float = 1.2e6

    # Dominant conductivity based on phonon or impurity scattering, in [1/s].
    # For details see Cumming et al. (2004) or Gourgouliatos and Cumming (2014).
    cfg["sigma"]: float = 1e24

    # Characteristic length scale of the magnetic field in [cm].
    cfg["L"]: float = 1e5

    # Characteristic electron density in [g/cm^3].
    cfg["n_e"]: float = 1e35

elif cfg["magneto-thermal_model"] == "SLy4_dip-tor_heavy":
    # Set the characteristic neutron star radius in [cm] for a mass of 1.4 Msun.
    cfg["NS_radius"]: float = 1.170e6

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

    # Set the path to where the magneto-thermal results are saved.
    cfg["magneto-thermal_path"]: str = str(
        pathlib.Path().joinpath(
            cfg["path_to_software"],
            "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/SLy4_dip-tor_heavy-envelope/",
        )
    )

elif (cfg["magneto-thermal_model"] == "BSk24_dip-tor_heavy") or (
    cfg["magneto-thermal_model"] == "BSk24_dip-tor_light"
):
    # Set the characteristic neutron star radius in [cm] for a mass of 1.4 Msun.
    cfg["NS_radius"]: float = 1.259e6

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

    # Set the path to where the magneto-thermal results are saved.
    if cfg["magneto-thermal_model"] == "BSk24_dip-tor_heavy":
        cfg["magneto-thermal_path"]: str = str(
            pathlib.Path().joinpath(
                cfg["path_to_software"],
                "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/BSk24_dip-tor_heavy-envelope/",
            )
        )
    elif cfg["magneto-thermal_model"] == "BSk24_dip-tor_light":
        cfg["magneto-thermal_path"]: str = str(
            pathlib.Path().joinpath(
                cfg["path_to_software"],
                "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/BSk24_dip-tor_light-envelope/",
            )
        )

elif (cfg["magneto-thermal_model"] == "BSk24_multi_heavy") or (
    cfg["magneto-thermal_model"] == "BSk24_multi_light"
):
    # Set the characteristic neutron star radius in [cm] for a mass of 1.4 Msun.
    cfg["NS_radius"]: float = 1.259e6

    # Power-law indices.
    cfg["a1"]: float = -0.15
    cfg["a2"]: float = -4.0

    # Timescale parameters, normalizations and power-law indices.
    cfg["A1"]: float = 9.0e13
    cfg["b1"]: float = -0.8
    cfg["A2"]: float = 6.0e8
    cfg["b2"]: float = -0.2

    # Timescale in [yr] when transitioning from the simulated curves to the simple late-time power-law evolution.
    cfg["tau_late"]: float = 2.0e6

    # Set the path to where the magneto-thermal results are saved.
    if cfg["magneto-thermal_model"] == "BSk24_multi_heavy":
        cfg["magneto-thermal_path"]: str = str(
            pathlib.Path().joinpath(
                cfg["path_to_software"],
                "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/BSk24_multi_heavy-envelope/",
            )
        )
    elif cfg["magneto-thermal_model"] == "BSk24_multi_light":
        cfg["magneto-thermal_path"]: str = str(
            pathlib.Path().joinpath(
                cfg["path_to_software"],
                "pypopsyn/simulator/magneto_rotational_physics/magneto-thermal_evol_curves/BSk24_multi_light-envelope/",
            )
        )
else:
    log.error(
        "The specified model for the magneto-thermal evolution is not supported."
        "Please choose between analytical, SLy4_dip-tor_heavy, BSk24_dip-tor_heavy, BSk24_dip-tor_light, BSk24_multi_heavy or"
        "BSk24_multi_light."
    )


# Late time power-law index (default value is taken from Pardo-Araujo et al. 2025).
cfg["a_late"]: float = -0.88

# Parameters for a log-normal distribution of the magnetic fields of the old millisecond pulsars.
cfg["B_millisec_mean"]: float = 8.5
cfg["B_millisec_sigma"]: float = 0.5

# ===================== RADIO EMISSION-MODEL PARAMETERS ========================

# Model for the radio beam aperture. Choose between "standard_period_cone" and "power-law_period_cone".
cfg["radio_beam_model"]: str = "standard_period_cone"

if cfg["radio_beam_model"] == "standard_period_cone":
    # Distance in [cm] from the center of the star to where the radio emission is generated (Johnston et al. 2020).
    cfg["r_em"]: float = 3.0e7

elif cfg["radio_beam_model"] == "power-law_period_cone":
    # Parameters taken from Maciesiak et al. (2012).
    cfg[
        "rho_b_0"
    ]: float = 2.5  # Half opening angle of the radio beam in [deg] corresponding to a spin period of 1 s.
    cfg["a_beam"]: float = -0.5  # Power-law exponent.

else:
    log.error(
        "The specified model for radio-beam geometry is not supported."
        "Please choose between standard_period_cone or power-law_period_cone."
    )

# Relevant parameters for the log-normally distributed luminosity, L.
# We have implemented two different prescriptions for the luminosity in the module
# pypopsyn/simulator/multiband_emission/emission_radio.py based on the luminosity depending on different parameters.
# One prescription follows Faucher-Giguère & Kaspi (2006) (lum_radio_ppdot model) and assumes that L depends on
# the period and period derivative.
# The second one assumes that L depends directly on the loss of rotational energy (lum_radio_edot model).

# Model pdf for the radio luminosity. Choose between "lum_radio_ppdot" and "lum_radio_edot".
cfg["radio_luminosity_model"]: str = "lum_radio_edot"

if cfg["radio_luminosity_model"] == "lum_radio_ppdot":
    # Parameters for the "lum_radio_ppdot" model (Graber et al. 2024).
    # These best parameters assume a mean_spectral_index = -1.6 (see Jankowski et al. 2018).
    cfg["L_radio_log10_mean"]: float = 35.5  # [erg s^(3 * epsilon_L - 1) ]
    cfg["L_radio_log10_sigma"]: float = 0.8
    cfg["epsilon_L"]: float = 0.5

elif cfg["radio_luminosity_model"] == "lum_radio_edot":
    # Parameters for the "lum_radio_edot" model (Pardo-Araujo et al. 2025).
    # These best parameters assume a mean_spectral_index = -1.8 (see Posselt et al. 2023).
    cfg["L_radio_log10_mean"]: float = 26.17  # [erg s^(- 1)]
    cfg["L_radio_log10_sigma"]: float = 0.8
    cfg["epsilon_L"]: float = 0.68
    cfg["Erot_dot_0"]: float = 1e29

else:
    log.error(
        "The specified model for radio luminosity is not supported."
        "Please choose between lum_radio_ppdot or lum_radio_edot."
    )

# Spectral index following a normal distribution as in Posselt et al. (2023). We set the standard deviation to 0 to
# efficiently produce a fixed spectral index.
cfg["mean_spectral_index"]: float = -1.8
cfg["std_spectral_index"]: float = 0

# Free electron density model for the Galaxy, choose between "ne2001" and "ymw16".
cfg["ed_model"]: str = "ymw16"

# ===================== RADIO DETECTION PARAMETERS ========================

# Information on the modeled radio surveys.
# To obtain the number of Galactic isolated neutron stars detected by the considered surveys,
# we removed extragalactic sources and those in globular clusters.
# To exclude recycled objects that we cannot model with our current framework,
# we also use a cut-off in period of P > 0.01s and period derivative of Pdot > 10^-19s/s.
# The latter however only applies to those objects with measured Pdot values,
# i.e., the counts below also include those pulsars with P > 0.01s that have no Pdot measurement.
cfg["surveys_radio"]: dict = {
    "PMPS": {
        "path": "pypopsyn/simulator/multiband_surveys/Parkes_parameters.json",
        "detected_real": 1045,
    },
    "SMPS": {
        "path": "pypopsyn/simulator/multiband_surveys/Swinburne_Parkes_parameters.json",
        "detected_real": 218,
    },
    "HTRU_low_mid": {
        "path_low": "pypopsyn/simulator/multiband_surveys/htru_low_parameters.json",
        "path_mid": "pypopsyn/simulator/multiband_surveys/htru_mid_parameters.json",
        "detected_real": 1095,
    },
    "HTRU_high": {
        "path": "pypopsyn/simulator/multiband_surveys/htru_high_parameters.json",
        "detected_real": 20,
    },
}

# Numbers of objects associated with the three pulsar surveys as followed up with the TPA programme on MeerKAT.
# For details see Posselt et al. (2023). Note these numbers are used in the pypopsyn/generator/generate_observed_data.py
# script and differ from those given in the full ATNF Pulsar Catalogue.
cfg["detected_meerkat_PMPS"]: int = 640
cfg["detected_meerkat_SMPS"]: int = 170
cfg["detected_meerkat_HTRU"]: int = 668

# ===================== X-RAY EMISSION-MODEL PARAMETERS ========================

# Array of abundances relative to hydrogen for 17 elements taken from table 2 in Wilms et al. (2000).
cfg["ISM_abundances"]: List[float] = [
    12.0,
    10.99,
    8.38,
    7.88,
    8.69,
    7.94,
    6.16,
    7.40,
    6.33,
    7.27,
    7.09,
    5.12,
    6.41,
    6.20,
    5.51,
    7.43,
    6.05,
]

# Enabling the use of an interpolator for the rate of crust failures to estimate the number of outburst events for
# neutron stars with different initial magnetic fields and ages. If set to False a simpler probabilistic approach
# will be used that takes into account the age but neglects the dependence of the outburst rate on the field strength
# and configuration. See the method outburst_filter_probabilistic in
# pypopsyn/simulator/multiband_emission/emission_xray.py for more details.
cfg["use_crust_failure_rate_interpolator"]: bool = True

# ===================== X-RAY DETECTION PARAMETERS ========================

# Information on the modeled X-ray surveys.
# To obtain the number of Galactic isolated neutron stars detected with a quiescent thermal emission in X-rays,
# we only include magnetars and XDINSs, as young RPPs with quiescent X-ray emission are primarily discovered through the
# detection of the supernova remnant emission and pulsar wind nebula contribution which we are not modeling in our code.
# We currently implement two different kinds of surveys in the X-rays. The first survey only takes into account a simple
# cut-off in flux. The second one tries to better match the observed distribution and considers that many magnetars have
# been discovered during an outburst phase. For those we consider a deeper survey to detect a quiescent emission. We
# then combine it with a shallower survey which detects only the brightest sources. Moreover, since we do not have full
# control over the observational biases for X-ray surveys, we try to match only the number of sources above a flux
# threshold of 2e-11 [erg s^-1 cm^-2] which defines the limit where we assume real surveys are complete (by looking at
# the logN-lgS distribution).
cfg["surveys_xray"]: dict = {
    "xray_flux_threshold": {
        "path": "pypopsyn/simulator/multiband_surveys/xray_flux_threshold_parameters.json",
        "detected_real": 5,
        "flux_threshold_completeness": 1.0e-11,
    },
    "xray_realistic": {
        "path": "pypopsyn/simulator/multiband_surveys/xray_realistic_parameters.json",
        "detected_real": 5,
        "flux_threshold_completeness": 1.0e-11,
    },
}


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
