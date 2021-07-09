"""
Models for the pulsars' gamma-ray emission geometry and luminosity.

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

NS_mass: float = cfg["NS_mass"]
NS_radius: float = cfg["NS_radius"]


def pdf_gamma_luminosity(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Draw random radio luminosities from a distribution that depends on the spin-down energy losses.
    We assume a random log-normal spread for the normalization constant L_0.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar gamma luminosity [erg s^(-1)] drawn from a log-normal distribution.
    """

    NS_number = len(P)

    # Canonical neutron star moment of inertia in [g cm^2] assuming a perfect solid sphere.
    NS_inertia = 2.0 / 5.0 * NS_mass * NS_radius ** 2

    # Calculate the rotational energy derivative in [erg / s].
    Erot_dot = NS_inertia * (2.0 * np.pi) ** 2 * P_dot / (P ** 3)

    L_0 = 10 ** np.random.normal(
        cfg["L_gamma_log10_mean"], cfg["L_gamma_log10_sigma"], NS_number
    )
    L_gamma = L_0 * Erot_dot ** cfg["epsilon"]

    return L_gamma


def erg_flux_gamma(L_gamma: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Compute the gamma flux observed here on Earth for each pulsars in [erg s^(-1) cm^(-2)].

    Args:
        L_gamma (np.ndarray): pulsar gamma luminosity in [erg s^(-1)].
        d (np.ndarray): distance from the pulsar in [kpc].

    Returns:
        (np.ndarray): observed pulsar gamma flux in [erg s^(-1) cm^(-2)].
    """

    # Convert distance from [kpc] to [cm].
    d_cm = d * const.KPC_TO_CM

    S_gamma = L_gamma / (4.0 * np.pi * d_cm ** 2)

    return S_gamma
