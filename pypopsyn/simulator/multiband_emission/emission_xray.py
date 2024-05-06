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
        (np.ndarray): temperature seen by a distant observer in [K].
    """
    R_obs = cfg["NS_radius"] / gr_correction
    T_obs = (Lx / (4 * np.pi * R_obs**2 * const.SIGMA_SB)) ** (1.0 / 4.0)

    return T_obs


def blackbody_intensity_spectrum(E: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Calculating the black-body intensity for a given temperature in [K].

    Args:
        E (np.ndarray): array of energies in [eV] where to compute the intensity.
        T (np.ndarray): array of surface temperatures in [K] for each neutron star.

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


def dirac_delta(x, epsilon=1e-10) -> np.ndarray:
    """
    Approximation of the Dirac delta function.
    """

    return np.where(np.abs(x) < epsilon, 2.0e-17, 0.0)


def n_plus(
    omega: np.ndarray,
    omega_0: np.ndarray,
    tau_0: np.ndarray,
    beta_T: np.ndarray,
) -> np.ndarray:
    """
    n+ function in eq. (36) in Lyutikov and Gavrill 2006.
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
    n- function in eq. (36) in Lyutikov and Gavrill 2006.
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
    Compute the transmitted and reflected flux probabilities (see Lyutikov and Gavrill 2006).
    """

    n_p = n_plus(omega, omega_0, tau_0, beta_T)
    n_m = n_minus(omega, omega_0, tau_0, beta_T)

    nm_norm_theory = (1 - np.exp(-tau_0)) / 2.0

    nm_norm_theory = nm_norm_theory[:, np.newaxis]

    nm_norm = trapz(n_m, omega, axis=1) / (nm_norm_theory)
    np_norm = trapz(n_p, omega, axis=1) / (1 - nm_norm_theory)

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
    n_source: np.ndarray,
) -> np.ndarray:
    """
    Compute the spectrum resulting from resonant cyclothron scattering given a source intensity spectrum (see Lyutikov and Gavrill 2006).
    """
    rcs_spectrum = np.zeros((len(tau_0), len(E)))

    # Convert photon energy in eV into omega frequencies in s^-1.
    freq_0 = E_0 * const.EV_TO_ERG / const.H
    omega_0 = 2.0 * np.pi * freq_0

    freq = E * const.EV_TO_ERG / const.H
    omega = 2.0 * np.pi * freq

    p_trans, p_refl = trans_reflect_prob(omega, omega_0, tau_0, beta_T)

    n = n_source[:, np.newaxis, :]
    n_trans = trapz((n * p_trans), omega_0, axis=2)
    print(n_trans.shape)
    rcs_spectrum = rcs_spectrum + n_trans

    for i in range(6):
        n_reflect = trapz((n * p_refl), omega_0, axis=2)
        n_reflect = n_reflect[:, np.newaxis, :]
        n_trans_refl = trapz((n_reflect * p_trans), omega_0, axis=2)
        rcs_spectrum = rcs_spectrum + n_trans_refl

        n = n_trans_refl[:, np.newaxis, :]

    return rcs_spectrum


def flux_xray_absorbed(
    Lx: np.ndarray, RA: np.ndarray, DEC: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """
    Compute the x-ray flux density assuming a black-body spectral shape for the thermal x-ray emission.

    Args:
        Lx (np.ndarray): X-ray luminosity in [erg/s].
        RA (np.ndarray): array of right ascension in [deg] defined between [-90, 90] deg.
        DEC (np.ndarray): array of right ascension in [deg] defined between [0, 360] deg.

    Returns:
        (np.ndarray): absorbed x-ray fluxes in [erg s^-1 cm^-2].
    """
    T_obs = T_from_Lx(Lx)

    R_obs = cfg["NS_radius"] / gr_correction

    # Define the energy range between 0.01 keV and 20 keV (a larger energy range than the one where the absorption
    # cross-section is defined, is required in order to compute the resonant cyclothron scattered spectrum).
    E = np.logspace(1.0, np.log10(20000), 1000)
    I_bb = blackbody_intensity_spectrum(E, T_obs)

    # Calculate the black-body spectrum in [ph cm^-2 s^-1 eV^-1 sterad^-1].
    n_bb = I_bb / (E * const.EV_TO_ERG)

    tau_res = np.array([10.0])
    tau_0 = tau_res / 2
    beta_T = np.array([0.3])
    rcs_spectrum_trans = resonant_cyclothron_scat_spectrum(
        E, E, tau_0, beta_T, n_bb
    )

    N_H = nhm.compute_NH(RA, DEC, d)
    # Reshape N_H to make it compatible for broadcasting.
    N_H = N_H.reshape(-1, 1)

    sigma_ISM = xabs.absorption_cross_section_tot(E, cfg["ISM_abundances"])

    absorb_factor = np.exp(-sigma_ISM * N_H)
    I_absorbed = absorb_factor * rcs_spectrum_trans

    I_absorbed_bolom = trapz(I_absorbed, E, axis=-1)
    flux = (R_obs / d) ** 2 * np.pi * I_absorbed_bolom

    return flux
