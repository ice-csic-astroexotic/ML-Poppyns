"""
    Models for the x-ray emission of thermally emitting neutron stars.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Tuple

import numpy as np
import scipy.special as scsp
from scipy.integrate import trapz

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.interstellar_medium.nh_model as nhm
import pypopsyn.simulator.interstellar_medium.xray_abs_cross_section as xabs
from pypopsyn.simulator.config_simulator import cfg

# General relativity correction factor that accounts for space-time curvature around a neutron star.
gr_correction = (
    1 - (2 * const.G * cfg["NS_mass"]) / (const.C**2 * cfg["NS_radius"])
) ** (1.0 / 2.0)


def T_from_Lx(Lx: np.ndarray) -> np.ndarray:
    """
    Calculating the temperature seen by a distant observer from the simulated X-ray luminosity,
    by assuming a blackbody emission. This temperature is an observed average temperature of the
    neutron star surface corrected for the space-time curvature.

    Args:
        Lx (np.ndarray): X-ray luminosity in [erg/s].

    Returns:
        (np.ndarray): array of temperature seen by a distant observer in [K].
    """

    R_obs = cfg["NS_radius"] / gr_correction
    T_obs = (Lx / (4 * np.pi * R_obs**2 * const.SIGMA_SB)) ** (1.0 / 4.0)

    return T_obs


def blackbody_intensity_spectrum(E: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Calculating the black-body intensity for a given temperature in [K].

    Args:
        E (np.ndarray): array of energies in [eV] where to compute the intensity.
        T (np.ndarray): array of temperatures in [K].

    Returns:
        (np.ndarray): Intensity of the black-body spectrum in [erg cm^-2 s^-1 eV^-1 sterad^-1] for every temperature.
                      This will have shape (len(T), len(E)).
    """
    # Reshape T to make it compatible for broadcasting.
    T = T[:, np.newaxis]

    E_erg = E * const.EV_TO_ERG
    exponent = E_erg / (const.K_B * T)
    # Clip the maximum reachable value of the exponent to 709 to avoid RuntimeWarning: overflow encountered in exp
    exponent_clipped = np.clip(exponent, None, 709)
    I_bb = (
        2.0
        / (const.H**3 * const.C**2)
        * E_erg**3
        / (np.exp(exponent_clipped) - 1)
    ) * const.EV_TO_ERG

    return I_bb


def n_plus_no_delta(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Transmission function n+ without the Dirac delta term in eq. (36) in Lyutikov and Gavrill (2006).
    When computing the transmitted flux the Dirac delta term will be added analytically in order to avoid computing
    it numerically.

    Args:
        E (np.ndarray): array of energies of the transimitted photons in [eV].
        E_0 (np.ndarray): array of energies of the source photons in [eV].
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n+ without the Dirac delta term.
                      This will have shape (NS_number, len(E), len(E_0)).
    """

    # Reshape the input arrays to make it compatible for broadcasting.
    E_0 = E_0[np.newaxis, np.newaxis, :]
    E = E[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    x = (E - E_0) / E_0

    # Calculate term_1 safely, Replace x=0 with NaN for safety and set sqrt_arg_1 to np.inf when x was 0.
    x_safe = np.where(x == 0, np.nan, x)
    sqrt_arg_1 = (4.0 * beta_T - x_safe) / x_safe
    sqrt_arg_1 = np.where(x == 0, np.inf, sqrt_arg_1)
    # Replace the argument of the sqrt with NaN when it is less than 0 for safety and compute term 1.
    sqrt_arg_1_safe = np.where(sqrt_arg_1 < 0, np.nan, sqrt_arg_1)
    term_1 = tau_0 / (8.0 * beta_T) * sqrt_arg_1_safe**0.5
    # Replace the NaN values in term_1 with 0.
    term_1 = np.nan_to_num(term_1, nan=0)

    # Replace the argument of the sqrt with NaN when it is less than 0 for safety and compute I1.
    sqrt_arg_2 = x * (4.0 * beta_T - x)
    sqrt_arg_2_safe = np.where(sqrt_arg_2 < 0, np.nan, sqrt_arg_2)
    I1 = scsp.i1(tau_0 / (4.0 * beta_T) * sqrt_arg_2_safe**0.5)
    I1 = np.nan_to_num(I1, nan=0)

    term_2 = term_1 * I1

    n_p = np.exp(-tau_0 / 2.0) / E_0 * term_2

    return n_p


def n_minus(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Reflection function n- in eq. (36) in Lyutikov and Gavrill (2006).
    Note that in the original paper this equation misses a factor 1/2 inside the modified Bessel function.

    Args:
        E (np.ndarray): array of energies of the transimitted photons in [eV].
        E_0 (np.ndarray): array of energies of the source photons in [eV].
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n-.
                      This will have shape (NS_number, len(E), len(E_0)).
    """

    # Reshape omega to make it compatible for broadcasting.
    E_0 = E_0[np.newaxis, np.newaxis, :]
    E = E[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    x = (E - E_0) / E_0

    sqrt_arg = (2.0 * beta_T - x) * (x + 2.0 * beta_T)
    sqrt_arg_safe = np.where(sqrt_arg < 0, np.nan, sqrt_arg)
    I0 = scsp.i0(tau_0 / (4.0 * beta_T) * sqrt_arg_safe**0.5)
    I0 = np.nan_to_num(I0, nan=0)

    n_m = tau_0 / (8.0 * beta_T * E_0) * np.exp(-tau_0 / 2.0) * I0

    return n_m


def resonant_cyclotron_scat_spectrum(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
    I_ph_source: np.ndarray,
    n_reflections: int,
) -> np.ndarray:
    """
    Compute the spectrum resulting from resonant cyclotron scattering (RCS) given a source photon intensity spectrum
    considering multiple reflections and transmissions (see eq. (42) in Lyutikov and Gavrill 2006).

    Args:
        E (np.ndarray): array of energies in [eV] of the transmitted intensity.
        E_0 (np.ndarray): array of energies in [eV] of the source intensity.
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.
        I_ph_source (np.ndarray): intensity of the source in [ph cm^-2 s^-1 eV^-1 sterad^-1].
        n_reflections (int): number of reflections (6 reflections guarantees convergence of the final spectrum,
                             see Lyutikov and Gavrill 2006).

    Returns:
        (np.ndarray): resonant cyclotron scattering spectrum intensity in [ph cm^-2 s^-1 eV^-1 sterad^-1].
    """
    rcs_spectrum = np.zeros((len(tau_0), len(E)))

    # Compute the transmission and reflection probabilities.
    n_trans_no_delta = n_plus_no_delta(E, E_0, tau_0, beta_T)
    p_refl = n_minus(E, E_0, tau_0, beta_T)

    # Compute the RCS spectrum by considering multiple reflections and transmissions.
    # (see eq. 42 in Lyutikov and Gavrill 2006).
    I_ph = I_ph_source[:, np.newaxis, :]
    # For the transmission probability we calculate the analytical integral of the term with the delta function and
    # add it to the numerical integral of the second term.
    exp_fact = np.exp(-tau_0 / 2.0)
    exp_fact = exp_fact[:, np.newaxis]
    I_ph_trans = I_ph_source * exp_fact + trapz(
        (I_ph * n_trans_no_delta), E_0, axis=2
    )
    rcs_spectrum = rcs_spectrum + I_ph_trans

    for i in range(n_reflections):
        I_ph_reflect = trapz((I_ph * p_refl), E_0, axis=2)
        I_ph_reflect_reshape = I_ph_reflect[:, np.newaxis, :]
        I_ph_trans_refl = I_ph_reflect * exp_fact + trapz(
            (I_ph_reflect_reshape * n_trans_no_delta), E_0, axis=2
        )
        rcs_spectrum = rcs_spectrum + I_ph_trans_refl

        I_ph = I_ph_trans_refl[:, np.newaxis, :]

    return rcs_spectrum


def beta_plasma(B: np.ndarray) -> np.ndarray:
    """
    Approximated relation between the average plasma thermal velocity and magnetic field strength
    (see eq. (4) in Gullon et al. 2015 and fig. 11 in Rea et al. 2008).

    Args:
        B (np.ndarray): array of magnetic field strength in [G].

    Returns:
        (np.ndarray): average plasma thermal velocity in units of the speed of light.
    """
    beta = 0.001 * np.ones(len(B))
    beta[B > 1.0e13] = 0.3

    return beta


def resonant_optical_depth(B: np.ndarray) -> np.ndarray:
    """
    Approximated relation between the resonant optical depth and the magnetic field strength
    (see eq. (3) in Gullon et al. 2015 and fig. 11 in Rea et al. 2008).

    Args:
        B (np.ndarray): array of magnetic field strength in [G].

    Returns:
        (np.ndarray): resonant optical depth values.
    """
    tau_res = 0.001 * np.ones(len(B))
    tau_res[B > 1.0e13] = B[B > 1.0e13] / 1.0e14

    return tau_res


def flux_xray_absorbed(
    Lx: np.ndarray,
    B: np.ndarray,
    RA: np.ndarray,
    DEC: np.ndarray,
    d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the x-ray flux density assuming a black-body spectral shape for the thermal x-ray emission.

    Args:
        Lx (np.ndarray): X-ray luminosity in [erg/s].
        B (np.ndarray): array of magnetic field strength in [G].
        RA (np.ndarray): array of right ascension in [deg] defined between [-90, 90] deg.
        DEC (np.ndarray): array of right ascension in [deg] defined between [0, 360] deg.
        d (np.ndarray): array of distances in kpc.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): absorbed x-ray fluxes in [erg s^-1 cm^-2].
    """
    T_obs = T_from_Lx(Lx)

    # Consider a hot-spot radius that is a fraction of the neutron star radius.
    # R_hot_spot = np.random.uniform(0.01, 1, len(Lx)) * cfg["NS_radius"]
    # R_obs = R_hot_spot / gr_correction
    R_obs = cfg["NS_radius"] / gr_correction

    d = d * const.KPC_TO_CM

    # Define the energy range between 0.01 keV and 20 keV (a larger energy range than the one where the absorption
    # cross-section is defined, is required in order to have a good approximation of the RCS spectrum).
    E = np.logspace(1.0, np.log10(20000), 200)
    I_bb = blackbody_intensity_spectrum(E, T_obs)
    # Compute the intensity in [ph cm^-2 s^-1 eV^-1 sterad^-1].
    I_ph_bb = I_bb / (E * const.EV_TO_ERG)

    # Estimate the parameters to compute the RCS spectrum.
    tau_res = resonant_optical_depth(B)
    tau_0 = tau_res / 2.0
    beta_T = beta_plasma(B)

    # Compute the RCS spectrum and convert it in [erg cm^-2 s^-1 eV^-1 sterad^-1].
    I_rcs = resonant_cyclotron_scat_spectrum(
        E, E, tau_0, beta_T, I_ph_bb, n_reflections=4
    ) * (E * const.EV_TO_ERG)

    # Estimate the N_H column density.
    N_H = nhm.compute_NH(RA, DEC, d)
    # Reshape N_H to make it compatible for broadcasting.
    N_H = N_H[:, np.newaxis]

    # Estimate the X-ray absorption cross section.
    sigma_ISM = xabs.absorption_cross_section_tot(E, cfg["ISM_abundances"])

    # Compute the absorbed intensity (see eq. (2) in Wilms et al. 2000).
    absorb_factor = np.exp(-sigma_ISM * N_H)
    I_absorbed = absorb_factor * I_rcs

    # Compute the total observed flux in the energy range [0.01, 10] keV.
    E_mask = E <= 10000
    I_absorbed_bolom = trapz(I_absorbed[:, E_mask], E[E_mask], axis=1)
    flux = (R_obs / d) ** 2 * np.pi * I_absorbed_bolom

    return flux, N_H
