"""
Models for the pulsars' radio beam geometry and luminosity.

Authors:

        Michele Ronchi (ronchi@ice.csic.es)

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
from pypopsyn.simulator.configuration import cfg


def beam_aperture(P: np.ndarray, r_em: float) -> np.ndarray:
    """
    Half angular aperture in [rad] of the radio beam as a function of the spin period.
    It can be derived by assuming that the radio beam width extends in the open
    field line region around the magnetic poles of the star.
    See eq. (3.29) in Lorimer and Kramer (2005) and eq. (2) in Johnston et al. (2020)
    for a derivation and discussion of this model.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        r_em (float): distance from the center of the star where the radio emission
                      is supposed to be generated [cm].

    Returns:
        (np.ndarray): half angular aperture of the radio beam in [rad].
    """

    rho_b = np.sqrt(9.0 * np.pi * r_em / (2.0 * const.c * P))

    return rho_b


def pulse_width(
    chi: np.ndarray, rho_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Formula to evaluate the pulse width in [rad] from the radio beam angular aperture.
    This assumes that the line of sight intercepts the radio beam with an angle beta with
    respect to the center of the beam, so that -rho_b < beta < rho_b, with rho the angular
    aperture of the beam. See eq. (1) in Maciesiak et al. (2011a) and eq. (3.26) in Lorimer & Kramer (2004).

    Args:
        chi (np.ndarray): array of inclination angles between the magnetic axis and the rotation axis [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angles of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of pulse widths in [rad].
    """

    # Compute the impact parameter, i.e., the angular distance between the LOS intercept and the center of the beam.
    beta = los - chi

    # Compute the quantity sin(w/4) where w is the pulse width.
    sin_w_4 = np.array(
        np.sqrt(
            (np.sin(rho_b / 2.0) ** 2 - np.sin(beta / 2) ** 2)
            / (np.sin(chi + beta) * np.sin(chi))
        )
    )
    # If sin(w/4) > 1 set w = 2*np.pi.
    sin_w_4[sin_w_4 > 1] = 1.0

    width = 4.0 * np.arcsin(sin_w_4)

    return width


def solid_angle_radio_beams(rho_b: np.ndarray) -> np.ndarray:
    """
    Total solid angle covered by the two radio beams as a function of the radio beam aperture.

    Args:
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].

    Returns:
        (np.ndarray): Total solid angle covered by the radio beams.
    """

    solid_angle = 4 * np.pi * (1 - np.cos(rho_b))

    return solid_angle


def beam_fraction(chi: np.ndarray, rho_b: np.ndarray) -> np.ndarray:
    """
    Fraction of solid angle covered by the radio beam in an entire pulsar rotation
    as a function of the inclination angle chi and the radio beam aperture.
    This is also the probability that the radio beam intercepts our line of sight.
    See Appendix in Emmering and Chevalier (1989).

    Args:
        chi (np.ndarray): array of inclination angles of the magnetic axis
        with respect to the rotation axis of the pulsars in [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].

    Returns:
        (np.ndarray): fraction of solid angle swept by the radio beam.
    """

    beam_frac = np.cos(np.maximum(0, (chi - rho_b))) - np.cos(
        np.minimum(np.pi / 2.0, (chi + rho_b))
    )

    return beam_frac


def los_intercept(
    chi: np.ndarray, rho_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Evaluate if the radio beam intercepts the line of sight (LOS), assuming random
    orientation of the LOS with respect to the rotation axis.

    Args:
        chi (np.ndarray): array of inclination angles of the pulsars in [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angle of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of boolean variables: true if the radio beam intercepts the LOS and false if not.
    """

    condition = (los > chi - rho_b) & (los < chi + rho_b)
    intercepted = np.array(condition)

    return intercepted


def pdf_luminosity_radio(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Draw random bolometric radio luminosities from a distribution that depends on the spin period
     and spin period derivative.
    We assume a random log-normal spread for the normalization constant L_0.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar radio luminosity [erg s^(-1)] drawn from a log-normal distribution.
    """

    NS_number = len(P)

    L_0 = 10 ** np.random.normal(
        cfg["L_radio_log10_mean"], cfg["L_radio_log10_sigma"], NS_number
    )
    L_radio = L_0 * (P ** (-3) * P_dot) ** cfg["epsilon_L"]

    return L_radio


def flux_radio(
    L_radio: np.ndarray, d: np.ndarray, solid_angle: np.ndarray,
) -> np.ndarray:
    """
    Compute the intrinsic bolometric mean radio flux for each pulsars in [erg s^(-1) cm^(-2)].

    Args:
        L_radio (np.ndarray): pulsar bolometric radio luminosity in [erg s^(-1)].
        d (np.ndarray): distance to the pulsar in [kpc].
        solid_angle (np.ndarray): solid angle covered by the radio beams in [sterad].

    Returns:
        (np.ndarray): observed pulsar radio flux in [erg s^(-1) cm^(-2)].
    """

    # Convert distance from [kpc] to [cm].
    d_cm = d * const.KPC_TO_CM

    S_radio = L_radio / (solid_angle * d_cm ** 2)

    return S_radio


def flux_density_radio(
    S_radio_bol: np.ndarray,
    f: float,
    spectral_index: float = -1.6,
    f_min: float = 1.0e7,
    f_max: float = 1.0e11,
) -> np.ndarray:
    """
    Compute the radio flux density at a given frequency f assuming a power law spectral shape for the radio emission.

    Args:
        S_radio_bol (np.ndarray): pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].
        f (float): frequency in [Hz] at which the radio luminosity has to be computed.
        spectral_index (float): spectral index of the radio emission, assuming a power-law spectrum.
        f_min (float): frequency lower limit of the radio emission spectrum [Hz].
        f_max (float): frequency upper limit of the radio emission spectrum [Hz].

    Returns:
        (np.ndarray): pulsar radio flux density in [Jy] at the frequency f.
    """

    S_radio_f = (
        (spectral_index + 1)
        * S_radio_bol
        / (f_max ** (spectral_index + 1) - f_min ** (spectral_index + 1))
        * f ** spectral_index
    ) / const.JY_TO_ERG

    return S_radio_f
