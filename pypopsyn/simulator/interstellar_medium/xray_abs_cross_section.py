"""
Models for cross section of the X-ray interstellar absorption.

We translate into python the modules in Fortran 77 written by Monika Balucinska-Church and Dan McCammon
"Photoelectric Absorption Cross Sections with Variable Abundances" Ap.J. 400, 699 (1992) available for download at
https://cdsarc.cds.unistra.fr/viz-bin/cat/VI/62#/browse.

The atomic absorption cross sections were taken from Henke et al. (1982). Polynomial fits have been made to the
atomic absorption cross sections in the energy range of 0.03 -- 10 keV for seventeen elements: hydrogen, helium,
carbon, nitrogen, oxygen, neon, sodium, magnesium, aluminium, silicon, sulphur, chlorine, argon, calcium, chromium,
iron and nickel.
The functions fit Henke's data points with a typical error of 2% and a maximum error of 7%, except for points below
40~eV for argon, calcium and sodium, where the errors are larger. The effective cross section per hydrogen atom for
a particular set of elemental abundances may be calculated from the individual cross sections.
For more detail see Monika Balucinska-Church and Dan McCammon "Photoelectric Absorption Cross Sections with Variable
Abundances" Ap.J. 400, 699 (1992).

All data (except for helium) are from:
B. L. Henke, P. Lee, T. J. Tanaka, R. L. Shimabukuro and B. K. Fujikawa, 1982, Atomic Data and Nuclear Data Tables,
vol 27, p 1.

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


def aluminium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for aluminum where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 72.78
    mask_2 = (E >= 72.78) & (E < 1559.9)
    mask_3 = E >= 1559.9

    x[mask_1] = (
        26.90487
        + (3.0 - 9.135221) * Elog[mask_1]
        + 1.175546 * Elog[mask_1] ** 2
    )
    x[mask_2] = (
        -38.1232
        + 29.5161 * Elog[mask_2]
        - 4.45416 * Elog[mask_2] ** 2
        + 0.226204 * Elog[mask_2] ** 3
    )
    x[mask_3] = (
        14.6897
        + 4.22743 * Elog[mask_3]
        - 0.344185 * Elog[mask_3] ** 2
        + 8.18542e-3 * Elog[mask_3] ** 3
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def argon(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for argon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 245.0
    mask_2 = (E >= 245.0) & (E < 3202.9)
    mask_3 = E >= 3202.9
    x[mask_1] = (
        -330.3509
        + (267.7433 + 3.0) * Elog[mask_1]
        - 78.90498 * (Elog[mask_1] ** 2)
        + 10.35983 * (Elog[mask_1] ** 3)
        - 0.5140201 * (Elog[mask_1] ** 4)
    )
    x[mask_2] = (
        -5.71870
        + 8.85812 * Elog[mask_2]
        - 0.307357 * (Elog[mask_2] ** 2)
        + 0.00169351 * (Elog[mask_2] ** 3)
        - 0.0138134 * (Elog[mask_2] ** 4)
        + 0.00120451 * (Elog[mask_2] ** 5)
    )
    x[mask_3] = (
        19.1905
        + 2.74276 * Elog[mask_3]
        - 0.164603 * (Elog[mask_3] ** 2)
        + 0.00165895 * (Elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)
    return mass_abs_coeff


def calcium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for calcium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 349.31
    mask_2 = (E >= 349.31) & (E < 4038.1)
    mask_3 = E >= 4038.1
    x[mask_1] = (
        -873.972
        + (865.5231 + 3.0) * Elog[mask_1]
        - 339.678 * (Elog[mask_1] ** 2)
        + 66.83369 * (Elog[mask_1] ** 3)
        - 6.590398 * (Elog[mask_1] ** 4)
        + 0.2601044 * (Elog[mask_1] ** 5)
    )
    x[mask_2] = (
        -3449.707
        + (2433.409 + 3.0) * Elog[mask_2]
        - 682.0668 * (Elog[mask_2] ** 2)
        + 95.3563 * (Elog[mask_2] ** 3)
        - 6.655018 * (Elog[mask_2] ** 4)
        + 0.1854492 * (Elog[mask_2] ** 5)
    )
    x[mask_3] = (
        18.89376
        + (3.0 - 0.2903538) * Elog[mask_3]
        - 0.1377201 * Elog[mask_3] ** 2
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def carbon(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for carbon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 284.0
    mask_2 = E >= 284.0
    x[mask_1] = (
        8.74161
        + (7.13348 * Elog[mask_1])
        + (-1.14604 * Elog[mask_1] ** 2)
        + (0.0677044 * Elog[mask_1] ** 3)
    )
    x[mask_2] = (
        3.81334
        + (8.93626 * Elog[mask_2])
        + (-1.06905 * Elog[mask_2] ** 2)
        + (0.0422195 * Elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def chlorine(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for chlorine where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 202.0
    mask_2 = (E >= 202.0) & (E < 2819.6)
    mask_3 = E >= 2819.6
    x[mask_1] = (
        6253.247
        + (3.0 - 8225.248) * Elog[mask_1]
        + 4491.675 * (Elog[mask_1] ** 2)
        - 1302.145 * (Elog[mask_1] ** 3)
        + 211.4881 * (Elog[mask_1] ** 4)
        - 18.25547 * (Elog[mask_1] ** 5)
        + 0.6545154 * (Elog[mask_1] ** 6)
    )
    x[mask_2] = (
        -233.0502
        + (143.9776 + 3.0) * Elog[mask_2]
        - 31.12463 * (Elog[mask_2] ** 2)
        + 2.938618 * (Elog[mask_2] ** 3)
        - 0.104096 * (Elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -23.74675
        + (14.50997 + 3.0) * Elog[mask_3]
        - 1.857953 * Elog[mask_3] ** 2
        + 6.6208832e-2 * (Elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def chromium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for chromium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 598.0
    mask_2 = (E >= 598.0) & (E < 691.0)
    mask_3 = (E >= 691.0) & (E < 5988.8)
    mask_4 = E >= 5988.8
    x[mask_1] = (
        -0.4919405
        + (12.66939 + 3.0) * Elog[mask_1]
        - 5.199775 * Elog[mask_1] ** 2
        + 1.086566 * (Elog[mask_1] ** 3)
        - 0.1196001 * (Elog[mask_1] ** 4)
        + 5.2152011e-3 * (Elog[mask_1] ** 5)
    )
    x[mask_2] = 27.29282 + (3.0 - 2.703336) * Elog[mask_2]
    x[mask_3] = (
        -15.2525
        + (13.23729 + 3.0) * Elog[mask_3]
        - 1.966778 * (Elog[mask_3] ** 2)
        + 8.062207e-2 * (Elog[mask_3] ** 3)
    )
    x[mask_4] = (
        8.307041
        + (2.008987 + 3.0) * Elog[mask_4]
        - 0.2580816 * (Elog[mask_4] ** 2)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def fano_resonance_line(
    a: float, b: float, c: float, lambd: np.ndarray
) -> np.ndarray:
    # Constants
    EPS = 911.2671 / lambd  # Energy in Rydbergs
    EPSI = 3.0 - 1.0 / (b * b) + 1.807317
    x = 2.0 * (EPS - EPSI) / c

    return (x - a) ** 2 / (1.0 + x**2)


def helium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for helium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    # Constants
    AV = 6.022045e23  # Avogadro's number
    AW = 4.0026  # Atomic weight hydrogen

    # Coefficients for polynomial
    C1 = [
        -2.953607e1,
        7.083061e0,
        8.678646e-1,
        -1.221932e0,
        4.052997e-2,
        1.317109e-1,
        -3.265795e-2,
        2.500933e-3,
    ]
    C2 = [
        -2.465188e1,
        4.354679e0,
        -3.553024e0,
        5.573040e0,
        -5.872938e0,
        3.720797e0,
        -1.226919e0,
        1.576657e-1,
    ]

    # Parameters for resonances
    Q = [2.81, 2.51, 2.45, 2.44]
    NU = [1.610, 2.795, 3.817, 4.824]
    GAMMA = [2.64061e-3, 6.20116e-4, 2.56061e-4, 1.320159e-4]

    # Start
    lambd = 12398.54 / E
    x = np.log10(lambd)
    y = np.zeros(len(lambd))

    mask_1 = lambd < 46.0
    mask_2 = (lambd >= 46.0) & (lambd < 503.97)

    y[mask_1] = sum([C2[i] * (x[mask_1] ** i) for i in range(len(C2))])
    y[mask_2] = sum([C1[i] * (x[mask_2] ** i) for i in range(len(C1))])
    y[mask_2] += sum(
        [
            np.log10(fano_resonance_line(Q[i], NU[i], GAMMA[i], lambd[mask_2]))
            for i in range(len(Q))
        ]
    )

    mass_abs_coeff = 10**y * AV / AW

    return mass_abs_coeff


def hydrogen(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for hydrogen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)

    x = (
        21.46941
        + (3.0 - 2.060152) * Elog
        - 0.1492932 * (Elog**2)
        + 5.4634294e-3 * (Elog**3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def iron(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for iron where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 707.4
    mask_2 = (E >= 707.4) & (E < 7111.2)
    mask_3 = E >= 7111.2

    x[mask_1] = (
        -15.07332
        + (18.94335 + 3.0) * Elog[mask_1]
        - 4.862457 * (Elog[mask_1] ** 2)
        + 0.5573765 * (Elog[mask_1] ** 3)
        - 3.0065542e-2 * (Elog[mask_1] ** 4)
        + 4.9834867e-4 * (Elog[mask_1] ** 5)
    )
    x[mask_2] = (
        -253.0979
        + (135.4238 + 3.0) * Elog[mask_2]
        - 25.47119 * Elog[mask_2] ** 2
        + 2.08867 * (Elog[mask_2] ** 3)
        - 6.4264648e-2 * (Elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -1.037655
        + (4.022304 + 3.0) * Elog[mask_3]
        - 0.3638919 * Elog[mask_3] ** 2
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def magnesium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for magnesium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 49.45
    mask_2 = (E >= 49.45) & (E < 1303.4)
    mask_3 = E >= 1303.4

    x[mask_1] = 7.107172 + (0.7359418 + 3.0) * Elog[mask_1]
    x[mask_2] = (
        -81.32915
        + (62.2775 + 3.0) * Elog[mask_2]
        - 15.00826 * Elog[mask_2] ** 2
        + 1.558686 * (Elog[mask_2] ** 3)
        - 6.1339621e-2 * (Elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -9.161526
        + (10.07448 + 3.0) * Elog[mask_3]
        - 1.435878 * Elog[mask_3] ** 2
        + 5.2728362e-2 * (Elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def neon(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for neon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 867.0
    mask_2 = E >= 867.0

    x[mask_1] = (
        -3.04041
        + (13.0071 * Elog[mask_1])
        + (-1.93205 * (Elog[mask_1] ** 2))
        + (0.0977639 * (Elog[mask_1] ** 3))
    )
    x[mask_2] = (
        17.6007
        + (3.29278 * Elog[mask_2])
        + (-0.263065 * Elog[mask_2] ** 2)
        + (5.68290e-3 * Elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def nickel(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for nickel where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 853.6
    mask_2 = (E >= 853.6) & (E < 8331.6)
    mask_3 = E >= 8331.6

    x[mask_1] = (
        -7.919931
        + (11.06475 + 3.0) * Elog[mask_1]
        - 1.935318 * (Elog[mask_1] ** 2)
        + 9.3929626e-2 * (Elog[mask_1] ** 3)
    )
    x[mask_2] = (
        3.71129
        + (8.45098 * Elog[mask_2])
        + (-0.896656 * Elog[mask_2] ** 2)
        + (0.0324889 * Elog[mask_2] ** 3)
    )
    x[mask_3] = 28.4989 + (0.485797 * Elog[mask_3])

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def nitrogen(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for nitrogen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 401.0
    mask_2 = E >= 401.0

    x[mask_1] = (
        9.24058
        + (7.02985 * Elog[mask_1])
        + (-1.08849 * Elog[mask_1] ** 2)
        + (0.0611007 * Elog[mask_1] ** 3)
    )
    x[mask_2] = (
        -13.0353
        + (15.4851 * Elog[mask_2])
        + (-1.89502 * Elog[mask_2] ** 2)
        + (0.0769412 * Elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def oxygen(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for oxygen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """
    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 531.7
    mask_2 = E >= 531.7

    x[mask_1] = (
        2.57264
        + (10.9321 * Elog[mask_1])
        + (-1.79383 * Elog[mask_1] ** 2)
        + (0.102619 * Elog[mask_1] ** 3)
    )
    x[mask_2] = (
        16.53869
        + (0.6428144 + 3.0) * Elog[mask_2]
        - 0.3177744 * (Elog[mask_2] ** 2)
        + 7.9471897e-3 * (Elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def silicon(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for silicon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 100.6
    mask_2 = (E >= 100.6) & (E < 1840.0)
    mask_3 = E >= 1840.0

    x[mask_1] = (
        -3.066295
        + (7.006248 + 3.0) * Elog[mask_1]
        - 0.9627411 * (Elog[mask_1] ** 2)
    )
    x[mask_2] = (
        -182.7217
        + (125.061 + 3.0) * Elog[mask_2]
        - 29.47269 * (Elog[mask_2] ** 2)
        + 3.03284 * (Elog[mask_2] ** 3)
        - 0.1173096 * (Elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -33.39074
        + (18.42992 + 3.0) * Elog[mask_3]
        - 2.385117 * (Elog[mask_3] ** 2)
        + 8.887583e-2 * (Elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def sodium(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for sodium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 1071.7
    mask_2 = E >= 1071.7

    x[mask_1] = (
        -2737.598
        + (2798.704 + 3.0) * Elog[mask_1]
        - 1009.892 * (Elog[mask_1] ** 2)
        + 87.16455 * (Elog[mask_1] ** 3)
        + 43.20644 * (Elog[mask_1] ** 4)
        - 15.27259 * (Elog[mask_1] ** 5)
        + 2.180531 * (Elog[mask_1] ** 6)
        - 0.1526546 * (Elog[mask_1] ** 7)
        + 4.3137977e-3 * (Elog[mask_1] ** 8)
    )
    x[mask_2] = (
        1.534019
        + (6.261744 + 3.0) * Elog[mask_2]
        - 0.9914126 * (Elog[mask_2] ** 2)
        + 3.5278253e-2 * (Elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def sulfur(E: np.ndarray) -> np.ndarray:
    """
    Calculates mass absorption coefficient (mu/rho) for sulfur where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    Elog = np.log(E)
    x = np.zeros_like(Elog)
    mask_1 = E < 165.0
    mask_2 = (E >= 165.0) & (E < 2470.5)
    mask_3 = E >= 2470.5

    x[mask_1] = (
        598.2911
        + (3.0 - 678.2265) * Elog[mask_1]
        + 308.1133 * (Elog[mask_1] ** 2)
        - 68.99324 * (Elog[mask_1] ** 3)
        + 7.62458 * (Elog[mask_1] ** 4)
        - 0.3335031 * (Elog[mask_1] ** 5)
    )
    x[mask_2] = (
        3994.831
        + (3.0 - 3693.886) * Elog[mask_2]
        + 1417.287 * (Elog[mask_2] ** 2)
        - 287.9909 * (Elog[mask_2] ** 3)
        + 32.70061 * (Elog[mask_2] ** 4)
        - 1.968987 * (Elog[mask_2] ** 5)
        + 4.9149349e-2 * (Elog[mask_2] ** 6)
    )
    x[mask_3] = (
        -22.49628
        + (14.24599 + 3.0) * Elog[mask_3]
        - 1.848444 * (Elog[mask_3] ** 2)
        + 6.6506132e-2 * (Elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (E**3)

    return mass_abs_coeff


def absorption_cross_section_tot(
    E: np.ndarray, abundances: np.ndarray
) -> np.ndarray:
    """
    Calculates the effective absorption cross section in units of cm^2/(hydrogen atom)
    at energy E in eV for the specified abundances of the elements.
    This function is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
        abundances (np.ndarray): array of abundances in log10 relative to hydrogen (= 12.00).
    Return:
        effective cross section (np.ndarray): effective cross section as a function of the energy
        in cm^2/(hydrogen atom).
    """
    # atomic weights of the elements
    atomic_weights = np.array(
        [
            1.00797,
            4.0026,
            12.01115,
            14.0067,
            15.9994,
            20.183,
            22.9898,
            24.312,
            26.9815,
            28.086,
            32.064,
            35.453,
            39.94,
            40.08,
            51.996,
            55.847,
            58.71,
        ]
    )
    av = 6.022045e23

    # Mass absorption cross sections for each element
    mass_abs_coeff = np.zeros((len(atomic_weights), len(E)))
    mass_abs_coeff[0] = hydrogen(E)
    mass_abs_coeff[1] = helium(E)
    mass_abs_coeff[2] = carbon(E)
    mass_abs_coeff[3] = nitrogen(E)
    mass_abs_coeff[4] = oxygen(E)
    mass_abs_coeff[5] = neon(E)
    mass_abs_coeff[6] = sodium(E)
    mass_abs_coeff[7] = magnesium(E)
    mass_abs_coeff[8] = aluminium(E)
    mass_abs_coeff[9] = silicon(E)
    mass_abs_coeff[10] = sulfur(E)
    mass_abs_coeff[11] = chlorine(E)
    mass_abs_coeff[12] = argon(E)
    mass_abs_coeff[13] = calcium(E)
    mass_abs_coeff[14] = chromium(E)
    mass_abs_coeff[15] = iron(E)
    mass_abs_coeff[16] = nickel(E)

    cross_section_tot = np.zeros_like(E)

    # Loop through elements
    for i in range(len(atomic_weights)):
        cross_section_tot += (
            atomic_weights[i]
            * mass_abs_coeff[i]
            / av
            * (10 ** (abundances[i] - abundances[0]))
        )

    return cross_section_tot


def absorption_cross_section_approx(E: np.ndarray) -> np.ndarray:
    """
    This function implements the approximation of Morrison and McCammon (1983)
    to the interstellar photoelectric absorption cross-section. Energy is in eV
    and the resultant cross-section is in cm**2/hydrogen atom. Abundances of other
    elements relative to hydrogen are appropriate for the interstellar medium in
    the solar neighborhood (see Table 1 in Morrison and McCammon 1983). This function
    is valid only over the energy range 30 - 10,000 eV.
    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
        abundances (np.ndarray): array of abundances in log10 relative to hydrogen (= 12.00).
    Return:
        effective cross section (np.ndarray): effective cross section as a function of the energy
        in cm^2/(hydrogen atom).
    """

    # Energy intervals in keV.
    Emax = np.array(
        [
            0.100,
            0.284,
            0.400,
            0.532,
            0.707,
            0.867,
            1.303,
            1.840,
            2.471,
            3.210,
            4.038,
            7.111,
            8.331,
            10.0,
        ]
    )

    # Polynomial coefficients.
    c0 = np.array(
        [
            17.3,
            34.6,
            78.1,
            71.4,
            95.5,
            308.9,
            120.6,
            141.3,
            202.7,
            342.7,
            352.2,
            433.9,
            629.0,
            701.2,
        ]
    )
    c1 = np.array(
        [
            608.1,
            267.9,
            18.8,
            66.8,
            145.8,
            -380.6,
            169.3,
            146.8,
            104.7,
            18.7,
            18.7,
            -2.4,
            30.9,
            25.2,
        ]
    )
    c2 = np.array(
        [
            -2150.0,
            -476.1,
            4.3,
            -51.4,
            -61.1,
            294.0,
            -47.7,
            -31.5,
            -17.0,
            0.0,
            0.0,
            0.75,
            0.0,
            0.0,
        ]
    )

    # Convert energy to keV.
    E_keV = E / 1e3

    # Find the energy interval.
    i = np.searchsorted(Emax, E_keV)
    # Handle the case where E_keV exceeds the maximum energy.
    i = np.minimum(i, len(Emax) - 1)

    # Calculate cross-section
    cross_section_ism = (
        (c0[i] + c1[i] * E_keV + c2[i] * E_keV**2) / (E_keV**3) * 1e-24
    )

    return cross_section_ism
