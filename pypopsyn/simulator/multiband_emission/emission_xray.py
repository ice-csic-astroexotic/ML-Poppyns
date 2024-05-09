"""
Models for the x-ray emission of thermally emitting neutron stars.

Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2024

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
import scipy.special as scsp
from scipy.integrate import trapz

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.interstellar_medium.nh_model as nhm
import pypopsyn.simulator.interstellar_medium.xray_abs_cross_section as xabs
from pypopsyn.simulator.configuration import cfg

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
    I_bb = (
        2.0
        / (const.H**3 * const.C**2)
        * E_erg**3
        / (np.exp(E_erg / (const.K_B * T)) - 1)
    ) * const.EV_TO_ERG

    return I_bb


def dirac_delta(x: np.ndarray) -> np.ndarray:
    """
    Approximation of the Dirac delta function.

    Args:
        x (np.ndarray): array of values.

    Returns:
        (np.ndarray): Value of the Dirac delta function.
    """
    epsilon = 1e-10

    # The value of the Dirac delta inside the interval [-epsilon, epsilon] is chosen to be 2.0e-17 to have
    # meaningful results when computing the RCS spectrum.
    dirac_delta = np.where(np.abs(x) < epsilon, 2.0e-17, 0.0)

    return dirac_delta


def n_plus(
    omega: np.ndarray,
    omega_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Transimission function n+ in eq. (36) in Lyutikov and Gavrill 2006.

    Args:
        omega (np.ndarray): array of frequencies (in the form 2*pi*f) for the transimitted photons in Hz.
        omega_0 (np.ndarray): array of frequencies (in the form 2*pi*f) for the source photons in Hz.
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. 2 in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n+.
                      This will have shape (NS_number, len(omega), len(omega_0)).
    """

    # Reshape the input arrays to make it compatible for broadcasting.
    omega_0 = omega_0[np.newaxis, np.newaxis, :]
    omega = omega[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    term_1 = (
        tau_0
        / (8.0 * beta_T * omega_0)
        * ((omega_0 * (1.0 + 4.0 * beta_T) - omega) / (omega - omega_0)) ** 0.5
    )
    I1 = scsp.i1(
        tau_0
        / (4.0 * beta_T * omega_0)
        * ((omega - omega_0) * (omega_0 * (1.0 + 4.0 * beta_T) - omega)) ** 0.5
    )

    term_2 = term_1 * I1
    term_2 = np.nan_to_num(term_2, nan=0)

    n_p = np.exp(-tau_0 / 2.0) * (dirac_delta(omega - omega_0) + term_2)

    return n_p


def n_minus(
    omega: np.ndarray,
    omega_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    Reflection function n- in eq. (36) in Lyutikov and Gavrill 2006.

    Args:
        omega (np.ndarray): array of frequencies (in the form 2*pi*f) for the reflected photons in Hz.
        omega_0 (np.ndarray): array of frequencies (in the form 2*pi*f) for the source photons in Hz.
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. 2 in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission function n-.
                      This will have shape (NS_number, len(omega), len(omega_0)).
    """

    # Reshape omega to make it compatible for broadcasting.
    omega_0 = omega_0[np.newaxis, np.newaxis, :]
    omega = omega[np.newaxis, :, np.newaxis]
    tau_0 = tau_0[:, np.newaxis, np.newaxis]
    beta_T = beta_T[:, np.newaxis, np.newaxis]

    I0 = scsp.i0(
        tau_0
        / (2.0 * beta_T * omega_0)
        * (
            (omega_0 * (1.0 + 2.0 * beta_T) - omega)
            * (omega - omega_0 * (1.0 - 2.0 * beta_T))
        )
        ** 0.5
    )

    n_m = tau_0 / (8.0 * beta_T * omega_0) * np.exp(-tau_0 / 2.0) * I0
    n_m = np.nan_to_num(n_m, nan=0)

    return n_m


def trans_reflect_prob(
    omega: np.ndarray,
    omega_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Compute the transmitted and reflected flux probabilities by normalizing the transmission and reflection functions
    (see Lyutikov and Gavrill 2006).

    Args:
        omega (np.ndarray): array of frequencies (in the form 2*pi*f) for the reflected photons in Hz.
        omega_0 (np.ndarray): array of frequencies (in the form 2*pi*f) for the source photons in Hz.
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. 2 in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.

    Returns:
        (np.ndarray): Value of the transmission and reflection probabilities.
                      This will have shape (NS_number, len(omega), len(omega_0)).
    """

    n_p = n_plus(omega, omega_0, tau_0, beta_T)
    n_m = n_minus(omega, omega_0, tau_0, beta_T)

    # Normalization of the n- function (total reflection probability, see eq. (37) in Lyutikov and Gavrill 2006).
    p_reflect_tot = (1 - np.exp(-tau_0)) / 2.0
    # Reshape p_reflect_tot to make it compatible for broadcasting.
    p_reflect_tot = p_reflect_tot[:, np.newaxis]

    # Compute the normalizations for n+ and n- to transform them into probabilities.
    np_norm = trapz(n_p, omega, axis=1) / (1 - p_reflect_tot)
    nm_norm = trapz(n_m, omega, axis=1) / p_reflect_tot
    np_norm = np_norm[:, np.newaxis, :]
    nm_norm = nm_norm[:, np.newaxis, :]

    p_trans = n_p / np_norm
    p_refl = n_m / nm_norm

    return p_trans, p_refl


def resonant_cyclothron_scat_spectrum(
    E: np.ndarray,
    E_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
    I_ph_source: np.ndarray,
    n_reflections: int,
) -> np.ndarray:
    """
    Compute the spectrum resulting from resonant cyclothron scattering (RCS) given a source photon intensity spectrum
    considering multiple reflections and transmissions (see eq. (42) in Lyutikov and Gavrill 2006).

    Args:
        E (np.ndarray): array of energies in [eV] of the transmitted intensity.
        E_0 (np.ndarray): array of energies in [eV] of the source intensity.
        tau_0 (np.ndarray): array of optical depths tau_0 (see eq. 2 in Lyutikov and Gavrill 2006).
        beta_T (np.ndarray): array of thermal velocities for the electrons/positrons in units of the speed of light.
        I_ph_source (np.ndarray): intensity of the source in [ph cm^-2 s^-1 eV^-1 sterad^-1].
        n_reflections (int): number of reflections (6 reflections guarantees convergence of the final spectrum,
                             see Lyutikov and Gavrill 2006).

    Returns:
        (np.ndarray): resonant cyclothron scattering spectrum intensity in [ph cm^-2 s^-1 eV^-1 sterad^-1].
    """
    rcs_spectrum = np.zeros((len(tau_0), len(E)))

    # Convert photon energy in eV into omega frequencies in Hz.
    freq_0 = E_0 * const.EV_TO_ERG / const.H
    omega_0 = 2.0 * np.pi * freq_0

    freq = E * const.EV_TO_ERG / const.H
    omega = 2.0 * np.pi * freq

    # Compute the transmission and reflection probabilities.
    p_trans, p_refl = trans_reflect_prob(omega, omega_0, tau_0, beta_T)

    # Compute the RCS spectrum by considering multiple reflections and transmissions.
    # (see eq. 42 in Lyutikov and Gavrill 2006).
    I_ph = I_ph_source[:, np.newaxis, :]
    I_ph_trans = trapz((I_ph * p_trans), omega_0, axis=2)
    rcs_spectrum = rcs_spectrum + I_ph_trans

    for i in range(n_reflections):
        I_ph_reflect = trapz((I_ph * p_refl), omega_0, axis=2)
        I_ph_reflect = I_ph_reflect[:, np.newaxis, :]
        I_ph_trans_refl = trapz((I_ph_reflect * p_trans), omega_0, axis=2)
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
) -> np.ndarray:
    """
    Compute the x-ray flux density assuming a black-body spectral shape for the thermal x-ray emission.

    Args:
        Lx (np.ndarray): X-ray luminosity in [erg/s].
        B (np.ndarray): array of magnetic field strength in [G].
        RA (np.ndarray): array of right ascension in [deg] defined between [-90, 90] deg.
        DEC (np.ndarray): array of right ascension in [deg] defined between [0, 360] deg.
        d (np.ndarray): array of distances in kpc.

    Returns:
        (np.ndarray): absorbed x-ray fluxes in [erg s^-1 cm^-2].
    """
    T_obs = T_from_Lx(Lx)

    R_obs = cfg["NS_radius"] / gr_correction

    d = d * const.KPC_TO_CM

    # Define the energy range between 0.01 keV and 20 keV (a larger energy range than the one where the absorption
    # cross-section is defined, is required in order to have a good approximation of the RCS spectrum).
    E = np.logspace(1.0, np.log10(20000), 1000)
    I_bb = blackbody_intensity_spectrum(E, T_obs)
    # Compute the intensity in [ph cm^-2 s^-1 eV^-1 sterad^-1].
    I_ph_bb = I_bb / (E * const.EV_TO_ERG)

    # Estimate the parameters to compute the RCS spectrum.
    tau_res = resonant_optical_depth(B)
    tau_0 = tau_res / 2.0
    beta_T = beta_plasma(B)

    # Compute the RCS spectrum and convert it in [erg cm^-2 s^-1 eV^-1 sterad^-1].
    I_rcs = resonant_cyclothron_scat_spectrum(
        E, E, tau_0, beta_T, I_ph_bb, n_reflections=6
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

    return flux
