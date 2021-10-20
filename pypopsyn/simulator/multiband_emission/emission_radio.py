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
    for a derivation and discussion on this model.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        r_em (float): distance from the center of the star where the radio emission
                      is supposed to be generated [cm].

    Returns:
        (np.ndarray): half angular aperture of the radio beam in [rad].
    """

    theta_b = np.sqrt(9.0 * np.pi * r_em / (2.0 * const.c * P))
    log_theta_b = np.log10(theta_b)
    dispersion = 0.15

    theta_b_rand = 10 ** np.random.normal(
        log_theta_b, dispersion, len(log_theta_b)
    )

    return theta_b_rand


def pulse_width(
    chi: np.ndarray, theta_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Formula to evaluate the pulse width in [rad] from the radio beam angular aperture.
    This assume that the line of sight intercept the radio beam with an angle beta with
    respect to the center of the beam, so that -rho < beta < rho, with rho the angular
    aperture of the beam. See eq. (1) in Maciesiak et al. (2011b).

    Args:
        chi (np.ndarray): array of inclination angles between the magnetic axis and the rotation axis [rad].
        theta_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angle of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of pulse width in [rad].
    """

    # Compute the angular distance between the LOS intercept and the center of the beam.
    beta = los - chi

    # Compute the quantity sin(w/4).
    sin_w_4 = np.array(
        np.sqrt(
            (np.sin(theta_b / 2.0) ** 2 - np.sin(beta / 2) ** 2)
            / (np.sin(chi + beta) * np.sin(chi))
        )
    )
    # If sin(w/4) > 1 set w = 2*np.pi.
    sin_w_4[sin_w_4 > 1] = np.ones(len(sin_w_4[sin_w_4 > 1]))

    width = 4.0 * np.arcsin(sin_w_4)

    return width


def beam_fraction(chi: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """
    Fraction of solid angle covered by the radio beam in an entire pulsar rotation
    as a function of the inclination angle chi of the magnetic axis
    with respect to the rotation axis and the radio beam aperture.
    This is also the probability that the radio beam intercept our line of sight.
    See Appendix in Emmering and Chevalier (1989).

    Args:
        chi (np.ndarray): array of inclination angles of the pulsars in [rad].
        theta_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].

    Returns:
        (np.ndarray): fraction of solid angle swept by the radio beam.
    """

    beam_frac = np.cos(np.maximum(0, (chi - theta_b))) - np.cos(
        np.minimum(np.pi / 2.0, (chi + theta_b))
    )

    return beam_frac


def los_intercept(
    chi: np.ndarray, theta_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Evaluate if the radio beam intercept the line of sight (LOS), assuming random
    orientation of the LOS with respect to the rotation axis.

    Args:
        chi (np.ndarray): array of inclination angles of the pulsars in [rad].
        theta_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angle of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of boolean variables: true if the radio beam intercept the LOS and false if not.
    """

    condition = (los > chi - theta_b) & (los < chi + theta_b)
    intercepted = np.array(condition)

    return intercepted


def pdf_radio_luminosity(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Draw random radio luminosities from a distribution that depends on the spin period and spin period derivative.
    We assume a random log-normal spread for the normalization constant L_0.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar radio luminosity [erg s^(-1) Hz^(-1)] drawn from a log-normal distribution.
    """

    NS_number = len(P)

    L_0 = 10 ** np.random.normal(
        cfg["L_radio_log10_mean"], cfg["L_radio_log10_sigma"], NS_number
    )
    L_radio = L_0 * P ** cfg["epsilon1"] * P_dot ** cfg["epsilon2"]

    return L_radio


def spindown_power(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Evaluate the spin-down power of pulsars.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar spin-down powers in [erg s^(-1)].
    """
    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect solid sphere.
    NS_inertia = (
        2.0 / 5.0 * cfg["NS_mass"] * const.M_SUN * cfg["NS_radius"] ** 2
    )

    # Evaluate the rotational power.
    Erot_dot = 4.0 * np.pi ** 2 * NS_inertia * P_dot * P ** (-3)

    return Erot_dot


def radio_efficiency(Erot_dot: np.ndarray) -> np.ndarray:
    """
    Evaluate the radio emission efficiency of pulsars as a function of their spin-down power.
    We assume an efficiency that goes as Erot_dot^(-0.9) as in Szary et al. 2014.

    Args:
        Erot_dot (np.ndarray): array of spin-down powers in [erg s^(-1)].

    Returns:
        (np.ndarray): pulsar radio emission efficiency in [Hz^(-1)].
    """

    eff_max = 1.0

    efficiency = eff_max * (1.0e30 / Erot_dot) ** (0.9)

    log_eff = np.log10(efficiency)
    sigma = 0.8

    eff_rand = 10 ** np.random.normal(log_eff, sigma, len(log_eff))
    eff_rand[eff_rand > eff_max] = np.zeros(len(eff_rand[eff_rand > eff_max]))

    return eff_rand


def radio_spectral_index(Erot_dot: np.ndarray) -> np.ndarray:
    """
    Evaluate radio spectral index from an empirical fit.

    Args:
       Erot_dot (np.ndarray): array of spin-down powers in [erg s^(-1)].

    Returns:
        (np.ndarray): radio spectral index.
    """
    alpha = -1.68 * (Erot_dot / 10 ** 32.6) ** 0.13
    alpha_disp = 0.9

    alpha_rand = np.random.normal(alpha, alpha_disp, len(Erot_dot))

    return alpha_rand


def radio_luminosity(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Evaluate radio luminosities assuming that it depends on the spin-down power and a radio efficiency.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar radio luminosity [erg s^(-1) Hz^(-1)].
    """
    Erot_dot = spindown_power(P, P_dot)
    eff_radio = radio_efficiency(Erot_dot)
    L_radio_tot = eff_radio * Erot_dot

    # alpha = radio_spectral_index(Erot_dot)
    alpha = -1.7

    nu_min = 1.0e7
    nu_max = 1.0e11
    nu_c = 1.374e9

    L_radio_1400 = (
        (1 + alpha)
        * L_radio_tot
        * nu_c ** alpha
        / (nu_max ** (1 + alpha) - nu_min ** (1 + alpha))
    )

    return L_radio_1400


def erg_flux_radio(
    L_radio_1400: np.ndarray,
    d: np.ndarray,
    beam_frac: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute the radio flux observed here on Earth for each pulsars in erg s^(-1) cm^(-2) Hz^(-1).

    Args:
        L_radio_1400 (np.ndarray): pulsar radio luminosity in [erg s^(-1) Hz^(-1)].
        d (np.ndarray): distance from the pulsar in [kpc].
        beam_frac (np.ndarray): beam fraction.
        w (np.ndarray): intrinsic pulse width in [rad].

    Returns:
        (np.ndarray): observed pulsar radio flux in erg s^(-1) cm^(-2) Hz^(-1).
    """

    # Convert distance from [kpc] to [cm].
    d_cm = d * const.KPC_TO_CM
    # Compute the duty cycle.
    duty_cycle = w / (2.0 * np.pi)

    S_radio = L_radio_1400 / (4.0 * np.pi * beam_frac * d_cm ** 2) * duty_cycle

    return S_radio
