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
    T = T.reshape(-1, 1)

    E_erg = E * const.EV_TO_ERG
    I_bb = (
        2.0
        / (const.H**3 * const.C**2)
        * E_erg**3
        / (np.exp(E_erg / (const.K_B * T)) - 1)
    )

    return I_bb


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

    # Define the energy range between 0.03 keV and 10 keV (where the absorption cross-section is defined).
    E = np.logspace(np.log10(30), 4.0, 1000)
    I_bb = blackbody_intensity_spectrum(E, T_obs)

    N_H = nhm.compute_NH(RA, DEC, d)
    # Reshape N_H to make it compatible for broadcasting.
    N_H = N_H.reshape(-1, 1)

    sigma_ISM = xabs.absorption_cross_section_tot(E, cfg["ISM_abundances"])

    absorb_factor = np.exp(-sigma_ISM * N_H)
    I_bb_absorbed = absorb_factor * I_bb

    I_bb_absorbed_bolom = trapz(I_bb_absorbed, E * const.EV_TO_ERG, axis=-1)
    flux = (R_obs / d) ** 2 * np.pi * I_bb_absorbed_bolom

    return flux
