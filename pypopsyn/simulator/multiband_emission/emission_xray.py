"""
    Models for the X-ray emission of thermally emitting neutron stars.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Optional, Tuple

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
    by assuming a blackbody emission and the Stefan-Boltzmann law. This temperature is an observed average
    temperature of the neutron star surface corrected for the space-time curvature.

    Args:
        Lx (np.ndarray): X-ray thermal luminosity in [erg/s].

    Returns:
        (np.ndarray): Array of temperature seen by a distant observer in [K].
    """
    R_obs = cfg["NS_radius"] / gr_correction
    T_obs = (Lx / (4 * np.pi * R_obs**2 * const.SIGMA_SB)) ** (1.0 / 4.0)

    return T_obs


def blackbody_intensity_spectrum(E: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Calculating the black-body intensity for a given temperature in [K].

    Args:
        E (np.ndarray): Array of energies in [erg] where to compute the intensity.
        T (np.ndarray): Array of temperatures in [K].

    Returns:
        (np.ndarray): Intensity of the black-body spectrum in [erg cm^-2 s^-1 erg^-1 sterad^-1] for every temperature.
            This will have shape (len(T), len(E)).
    """
    # Reshape T to make it compatible for broadcasting.
    T = T[:, np.newaxis]

    exponent = E / (const.K_B * T)

    # Fix the maximum reachable value of the exponent to 709 to avoid RuntimeWarning: overflow encountered in exp.
    # The number 709 is the threshold number after which the warning appears.
    exponent_clipped = np.clip(exponent, None, 709)

    # Compute the black-body intensity in [erg cm^-2 s^-1 erg^-1 sterad^-1].
    I_bb = (
        2.0
        / (const.H**2 * const.C**2)
        * E**3
        / (np.exp(exponent_clipped) - 1)
    )

    # Since we later integrate directly in energy we correct the intensity for the Jacobian of the differential term
    # dE = h df.
    I_bb = I_bb / const.H

    return I_bb


def n_plus_without_delta(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Transmission function p+ in eq. (11) in overleaf without the Dirac delta term (see also the function n+ in eq. (35)
    in Lyutikov and Gavriil 2006).
    When computing the transmitted flux the Dirac delta term will be added analytically in order to avoid computing
    it numerically.

    Args:
        E (np.ndarray): Array of energies of the transmitted photons in [erg].
        E_0 (np.ndarray): Array of energies of the source photons in [erg].
        tau_0 (np.ndarray): Array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavriil 2006).
        beta_T (np.ndarray): Array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n+ without the Dirac delta term.
            This will have shape (NS_number, len(E), len(E_0)).
    """

    # Reshape the input arrays to make them compatible for broadcasting.
    E_0 = E_0[np.newaxis, np.newaxis, :]
    E = E[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    # Note that eta is a matrix with shape (1, len(E), len(E_0)).
    eta = (E - E_0) / E_0

    # In order to avoid warnings, such as `overflows`, `divide by zero` or `incorrect value in sqrt`,
    # when calculating the different contributions in the transmission coefficient, we replace eta=0
    # with NaN for safety and set sqrt_arg_1 to np.inf when eta was 0.
    eta_safe = np.where(eta == 0, np.nan, eta)
    sqrt_arg_1 = (4.0 * beta_T - eta_safe) / eta_safe
    sqrt_arg_1 = np.where(eta == 0, np.inf, sqrt_arg_1)

    # Replace the argument of the first sqrt with NaN when it is less than 0 for safety and compute term 1.
    sqrt_arg_1_safe = np.where(sqrt_arg_1 < 0, np.nan, sqrt_arg_1)
    term_1 = tau_0 / (8.0 * beta_T) * sqrt_arg_1_safe**0.5

    # Replace the NaN values and np.inf values in term_1 with 0 and a very large number respectively.
    term_1 = np.nan_to_num(term_1, nan=0)

    # Replace the argument of the second sqrt with NaN when it is less than 0 for safety and compute I1.
    sqrt_arg_2 = eta * (4.0 * beta_T - eta)
    sqrt_arg_2_safe = np.where(sqrt_arg_2 < 0, np.nan, sqrt_arg_2)
    I1 = scsp.i1(tau_0 / (4.0 * beta_T) * sqrt_arg_2_safe**0.5)
    I1 = np.nan_to_num(I1, nan=0)

    term_2 = term_1 * I1

    # Here we divide by E_0 to take into account the differential and rescale to the energy prescription.
    n_p = np.exp(-tau_0 / 2.0) / E_0 * term_2

    return n_p


def n_minus(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Reflection function p- in eq. (12) in overleaf (see also the function n- in eq. (35) in Lyutikov and Gavriil 2006).
    Note that in the original paper this equation misses a factor 1/2 and in the exponential term should be tau_0/2
    instead of tau_0 in the prefactor.

    Args:
        E (np.ndarray): Array of energies of the transmitted photons in [erg].
        E_0 (np.ndarray): Array of energies of the source photons in [erg].
        tau_0 (np.ndarray): Array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavriil 2006).
        beta_T (np.ndarray): Array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n-.
            This will have shape (NS_number, len(E), len(E_0)).
    """

    # Reshape the input arrays to make them compatible for broadcasting.
    E_0 = E_0[np.newaxis, np.newaxis, :]
    E = E[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    # Note that xi is a matrix with shape (1, len(E), len(E_0)).
    xi = (E_0 - E) / E_0

    # In order to avoid warnings, such as `incorrect value in sqrt` when calculating the different
    # contributions in the reflection coefficient, we replace the argument of the sqrt
    # with NaN when it is less than 0 for safety
    sqrt_arg = (2.0 * beta_T - xi) * (xi + 2.0 * beta_T)
    sqrt_arg_safe = np.where(sqrt_arg < 0, np.nan, sqrt_arg)
    I0 = scsp.i0(tau_0 / (4.0 * beta_T) * sqrt_arg_safe**0.5)
    I0 = np.nan_to_num(I0, nan=0)

    # Here we divide by E_0 to take into account the differential and rescale to the energy prescription.
    n_m = tau_0 / (8.0 * beta_T * E_0) * np.exp(-tau_0 / 2.0) * I0

    return n_m


def resonant_cyclotron_scat_spectrum(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
    I_ph_source: np.ndarray,
    n_reflections: Optional[int] = 6,
) -> np.ndarray:
    """
    Compute the spectrum resulting from resonant cyclotron scattering (RCS) given a source photon intensity spectrum
    considering multiple reflections and transmissions (see eq. (15) in overleaf and eq. (42) in Lyutikov and Gavriil 2006).

    Args:
        E (np.ndarray): Array of energies in [erg] of the transmitted intensity.
        E_0 (np.ndarray): Array of energies in [erg] of the source intensity.
        tau_0 (np.ndarray): Array of optical depths tau_0 (see eq. (2) in Lyutikov and Gavriil 2006).
        beta_T (np.ndarray): Array of thermal velocities for the electrons/positrons in units of the speed of light.
        I_ph_source (np.ndarray): Intensity of the source in [photon count cm^-2 s^-1 erg^-1 sterad^-1].
        n_reflections (int): Number of reflections (6 reflections guarantee good convergence of the final spectrum,
            see Lyutikov and Gavriil 2006).

    Returns:
        (np.ndarray): Resonant cyclotron scattering spectrum intensity in [photon count cm^-2 s^-1 erg^-1 sterad^-1] with
            shape (NS_number, len(E)).
    """
    rcs_spectrum = np.zeros((len(tau_0), len(E)))

    # Compute the transmission and reflection probabilities.
    n_trans_without_delta = n_plus_without_delta(E, E_0, tau_0, beta_T)
    p_refl = n_minus(E, E_0, tau_0, beta_T)

    # Compute the RCS spectrum by considering multiple reflections and transmissions.
    # (see eq. 42 in Lyutikov and Gavriil 2006).
    I_ph = I_ph_source[:, np.newaxis, :]
    # For the transmission probability we calculate the analytical integral of the term with the delta function and
    # add it to the numerical integral of the second term.
    exp_fact = np.exp(-tau_0 / 2.0)
    exp_fact = exp_fact[:, np.newaxis]
    I_ph_trans = I_ph_source * exp_fact + trapz(
        (I_ph * n_trans_without_delta), E_0, axis=2
    )
    rcs_spectrum = rcs_spectrum + I_ph_trans

    for i in range(n_reflections):
        I_ph_reflect = trapz((I_ph * p_refl), E_0, axis=2)
        I_ph_reflect_reshape = I_ph_reflect[:, np.newaxis, :]
        I_ph_trans_refl = I_ph_reflect * exp_fact + trapz(
            (I_ph_reflect_reshape * n_trans_without_delta), E_0, axis=2
        )
        rcs_spectrum = rcs_spectrum + I_ph_trans_refl

        I_ph = I_ph_trans_refl[:, np.newaxis, :]

    return rcs_spectrum


def beta_plasma(B: np.ndarray) -> np.ndarray:
    """
    Approximated relation between the average plasma thermal velocity and magnetic field strength
    (see eq. (4) in Gullón et al. 2015 and fig. 11 in Rea et al. 2008).

    Args:
        B (np.ndarray): Array of magnetic field strengths in [G].

    Returns:
        (np.ndarray): Average plasma thermal velocity in units of the speed of light.
    """
    beta = 0.001 * np.ones(len(B))
    beta[B > 1.0e13] = 0.3

    return beta


def resonant_optical_depth(B: np.ndarray) -> np.ndarray:
    """
    Approximated relation between the resonant optical depth and the magnetic field strength
    (see eq. (3) in Gullón et al. 2015 and fig. 11 in Rea et al. 2008).

    Args:
        B (np.ndarray): Array of magnetic field strengths in [G].

    Returns:
        (np.ndarray): Resonant optical depth values.
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
    Compute the X-ray observed absorbed flux assuming a black-body spectral shape for the thermal X-ray emission.

    Args:
        Lx (np.ndarray): X-ray luminosity in [erg/s].
        B (np.ndarray): Array of magnetic field strengths in [G].
        RA (np.ndarray): Array of right ascensions in [deg] defined between [-90, 90] deg.
        DEC (np.ndarray): Array of declinations in [deg] defined between [0, 360] deg.
        d (np.ndarray): Array of distances in kpc.

    Returns:
        (np.ndarray, np.ndarray): A tuple containing the following arrays:
            - absorbed x-ray fluxes in [erg s^-1 cm^-2].
            - value of the hydrogen column density in [cm^-2].
    """
    T_obs = T_from_Lx(Lx)

    R_obs = cfg["NS_radius"] / gr_correction

    d = d * const.KPC_TO_CM

    # Define the energy range between 0.01 keV and 20 keV. Note that we require a larger energy range than
    # the one used to determine the absorption cross-section in order to properly approximate the RCS spectrum.
    E = np.logspace(1.0, np.log10(20000), 1000)
    # Convert the energy array from [eV] to [erg].
    E_erg = E * const.EV_TO_ERG

    # Compute the black-body intensity in [erg cm^-2 s^-1 erg^-1 sterad^-1].
    I_bb = blackbody_intensity_spectrum(E_erg, T_obs)
    # Convert the intensity from [erg cm^-2 s^-1 erg^-1 sterad^-1] to [photon count cm^-2 s^-1 erg^-1 sterad^-1].
    I_ph_bb = I_bb / E_erg

    # Estimate the parameters to compute the RCS spectrum.
    tau_res = resonant_optical_depth(B)
    tau_0 = tau_res / 2.0
    beta_T = beta_plasma(B)

    # Compute the RCS intensity spectrum and convert it in [photon count cm^-2 s^-1 erg^-1 sterad^-1].
    # 6 reflections guarantee good convergence of the final spectrum see Lyutikov and Gavriil (2006).
    I_ph_rcs = resonant_cyclotron_scat_spectrum(
        E_erg, E_erg, tau_0, beta_T, I_ph_bb, n_reflections=6
    )
    # Convert the intensity from [photon count cm^-2 s^-1 erg^-1 sterad^-1] to [erg cm^-2 s^-1 erg^-1 sterad^-1] in order
    # to obtain the spectrum in terms of energy.
    I_rcs = I_ph_rcs * E_erg
    # Convert the intensity from [erg cm^-2 s^-1 erg^-1 sterad^-1] to [erg cm^-2 s^-1 eV^-1 sterad^-1].
    I_rcs = I_rcs * const.EV_TO_ERG

    # Estimate the N_H column density.
    N_H = nhm.compute_NH(RA, DEC, d)
    # Reshape N_H to make it compatible for broadcasting.
    N_H = N_H[:, np.newaxis]

    # Estimate the X-ray absorption cross section.
    sigma_ISM = xabs.absorption_cross_section_tot(E, cfg["ISM_abundances"])

    # Compute the absorbed intensity (see eq. (2) in Wilms et al. 2000).
    absorb_factor = np.exp(-sigma_ISM * N_H)
    I_absorbed = absorb_factor * I_rcs

    # Compute the total observed flux in the energy range [0.01, 10] keV (see eq. (17) in overleaf).
    E_mask = E <= 10000
    I_absorbed_bolom = trapz(I_absorbed[:, E_mask], E[E_mask], axis=1)
    flux = (R_obs / d) ** 2 * np.pi * I_absorbed_bolom

    return flux, N_H
