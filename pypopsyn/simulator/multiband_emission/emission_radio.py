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

    return theta_b


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


def pdf_radio_luminosity_lognorm(NS_number: int) -> np.ndarray:
    """
    Draw random radio luminosities from a log-normal distribution.

    Args:
        NS_number (int): total number of neutron stars created in the simulation.


    Returns:
        (np.ndarray): pulsar radio luminosity [erg s^(-1) Hz^(-1)] drawn from a log-normal distribution.
    """

    L_radio = 10 ** np.random.normal(
        cfg["L_radio_log10_mean"], cfg["L_radio_log10_sigma"], NS_number
    )

    return L_radio


def erg_flux_radio(
    L_radio: np.ndarray, d: np.ndarray, beam_frac: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """
    Compute the radio flux observed here on Earth for each pulsars in erg s^(-1) cm^(-2) Hz^(-1).

    Args:
        L_radio (np.ndarray): pulsar radio luminosity in [erg s^(-1) Hz^(-1)].
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

    S_radio = L_radio / (4.0 * np.pi * beam_frac * d_cm ** 2) * duty_cycle

    return S_radio
