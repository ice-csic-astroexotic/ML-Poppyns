"""
Models for cross section of the X-ray interstellar absorption.

This module translates the Fortran 77 modules written by Monika Balucinska-Church and Dan McCammon into Python.
For details on these routines see their paper "Photoelectric Absorption Cross Sections with Variable Abundances",
ApJ 400, 699 (1992), which is based on atomic absorption cross sections from Henke et al. (1982).
The routines themselves are available for download at https://cdsarc.cds.unistra.fr/viz-bin/cat/VI/62#/browse.

The polynomial fits in Balucinska-Church and McCammon (1992) are applicable to the atomic absorption cross sections
in the energy range of 0.03 -- 10 keV for seventeen elements: hydrogen, helium, carbon, nitrogen, oxygen, neon, sodium,
magnesium, aluminium, silicon, sulphur, chlorine, argon, calcium, chromium, iron and nickel.

The functions fit Henke's data points with a typical error of 2% and a maximum error of 7%, except for points below
~40eV for argon, calcium and sodium, where the errors are larger. The effective cross section per hydrogen atom for
a particular set of elemental abundances may be calculated from the individual cross sections. For more detail see
Balucinska-Church and McCammon (1992)

The underlying data (except for helium) are from: B. L. Henke, P. Lee, T. J. Tanaka, R. L. Shimabukuro and
B. K. Fujikawa, Atomic Data and Nuclear Data Tables, 27, 1 (1982)
The mass absorption coefficients for helium are in better agreement with the best experiments as well as theoretical
models (see W. F. Chen, G. Cooper, and C. E. Brion, Phys. Rev. A, 44, 186 (1991)).

Finally note that the cross sections here only take into account the neutral atomic form of the elements and do not
account for the possibility of ionization and the presence of molecules and grains. However, including these effects
would give only a minor correction to the cross sections (see Wilms, Allen, McCray 2000).


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

import pypopsyn.simulator.basics.constants as const

# Atomic weights for the seventeen elements.
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


def aluminium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for aluminum where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 72.78
    mask_2 = (energy >= 72.78) & (energy < 1559.9)
    mask_3 = energy >= 1559.9

    x[mask_1] = (
        26.90487
        + (3.0 - 9.135221) * elog[mask_1]
        + 1.175546 * elog[mask_1] ** 2
    )
    x[mask_2] = (
        -38.1232
        + 29.5161 * elog[mask_2]
        - 4.45416 * elog[mask_2] ** 2
        + 0.226204 * elog[mask_2] ** 3
    )
    x[mask_3] = (
        14.6897
        + 4.22743 * elog[mask_3]
        - 0.344185 * elog[mask_3] ** 2
        + 8.18542e-3 * elog[mask_3] ** 3
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def argon(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for argon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(energy)
    mask_1 = energy < 245.0
    mask_2 = (energy >= 245.0) & (energy < 3202.9)
    mask_3 = energy >= 3202.9
    x[mask_1] = (
        -330.3509
        + (267.7433 + 3.0) * elog[mask_1]
        - 78.90498 * (elog[mask_1] ** 2)
        + 10.35983 * (elog[mask_1] ** 3)
        - 0.5140201 * (elog[mask_1] ** 4)
    )
    x[mask_2] = (
        -5.71870
        + 8.85812 * elog[mask_2]
        - 0.307357 * (elog[mask_2] ** 2)
        + 0.00169351 * (elog[mask_2] ** 3)
        - 0.0138134 * (elog[mask_2] ** 4)
        + 0.00120451 * (elog[mask_2] ** 5)
    )
    x[mask_3] = (
        19.1905
        + 2.74276 * elog[mask_3]
        - 0.164603 * (elog[mask_3] ** 2)
        + 0.00165895 * (elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)
    return mass_abs_coeff


def calcium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for calcium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 349.31
    mask_2 = (energy >= 349.31) & (energy < 4038.1)
    mask_3 = energy >= 4038.1
    x[mask_1] = (
        -873.972
        + (865.5231 + 3.0) * elog[mask_1]
        - 339.678 * (elog[mask_1] ** 2)
        + 66.83369 * (elog[mask_1] ** 3)
        - 6.590398 * (elog[mask_1] ** 4)
        + 0.2601044 * (elog[mask_1] ** 5)
    )
    x[mask_2] = (
        -3449.707
        + (2433.409 + 3.0) * elog[mask_2]
        - 682.0668 * (elog[mask_2] ** 2)
        + 95.3563 * (elog[mask_2] ** 3)
        - 6.655018 * (elog[mask_2] ** 4)
        + 0.1854492 * (elog[mask_2] ** 5)
    )
    x[mask_3] = (
        18.89376
        + (3.0 - 0.2903538) * elog[mask_3]
        - 0.1377201 * elog[mask_3] ** 2
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def carbon(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for carbon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 284.0
    mask_2 = energy >= 284.0
    x[mask_1] = (
        8.74161
        + (7.13348 * elog[mask_1])
        + (-1.14604 * elog[mask_1] ** 2)
        + (0.0677044 * elog[mask_1] ** 3)
    )
    x[mask_2] = (
        3.81334
        + (8.93626 * elog[mask_2])
        + (-1.06905 * elog[mask_2] ** 2)
        + (0.0422195 * elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def chlorine(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for chlorine where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 202.0
    mask_2 = (energy >= 202.0) & (energy < 2819.6)
    mask_3 = energy >= 2819.6
    x[mask_1] = (
        6253.247
        + (3.0 - 8225.248) * elog[mask_1]
        + 4491.675 * (elog[mask_1] ** 2)
        - 1302.145 * (elog[mask_1] ** 3)
        + 211.4881 * (elog[mask_1] ** 4)
        - 18.25547 * (elog[mask_1] ** 5)
        + 0.6545154 * (elog[mask_1] ** 6)
    )
    x[mask_2] = (
        -233.0502
        + (143.9776 + 3.0) * elog[mask_2]
        - 31.12463 * (elog[mask_2] ** 2)
        + 2.938618 * (elog[mask_2] ** 3)
        - 0.104096 * (elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -23.74675
        + (14.50997 + 3.0) * elog[mask_3]
        - 1.857953 * elog[mask_3] ** 2
        + 6.6208832e-2 * (elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def chromium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for chromium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 598.0
    mask_2 = (energy >= 598.0) & (energy < 691.0)
    mask_3 = (energy >= 691.0) & (energy < 5988.8)
    mask_4 = energy >= 5988.8
    x[mask_1] = (
        -0.4919405
        + (12.66939 + 3.0) * elog[mask_1]
        - 5.199775 * elog[mask_1] ** 2
        + 1.086566 * (elog[mask_1] ** 3)
        - 0.1196001 * (elog[mask_1] ** 4)
        + 5.2152011e-3 * (elog[mask_1] ** 5)
    )
    x[mask_2] = 27.29282 + (3.0 - 2.703336) * elog[mask_2]
    x[mask_3] = (
        -15.2525
        + (13.23729 + 3.0) * elog[mask_3]
        - 1.966778 * (elog[mask_3] ** 2)
        + 8.062207e-2 * (elog[mask_3] ** 3)
    )
    x[mask_4] = (
        8.307041
        + (2.008987 + 3.0) * elog[mask_4]
        - 0.2580816 * (elog[mask_4] ** 2)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def fano_resonance_line(
    q: float, nu: float, gamma: float, lambd: np.ndarray
) -> np.ndarray:
    """
    Model for a Fano line profile that arise in case of resonant absorption.
    This line shape is taken from Fernley, Taylor and Seaton (1987).

    Args:
        q (float): Q coefficient for resonance (Fernley et al. 1987).
        nu (float): nu coefficient for resonance (Oza 1986).
        gamma (float): gamma coefficient for resonance (Oza 1986).
        lambd (np.ndarray): array of wavelengths in angstroms.

    Return:
        fano line profile (np.ndarray): fano line profile.
    """

    # Convert wavelength in angstrom into energy in Rydberg.
    e_ryd = (const.H * const.C) / (lambd / const.CM_TO_A) / const.RYD_TO_ERG

    epsi = 3.0 - 1.0 / (nu**2) + 1.807317

    x = 2.0 * (e_ryd - epsi) / gamma

    fano = (x - q) ** 2 / (1.0 + x**2)

    return fano


def helium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for helium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.
    This function is in better agreement with the best experiments as well as theoretical models
    (see W. F. Chen, G. Cooper, and C. E. Brion, Phys. Rev. A, 44, 186 (1991)).

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    He_atomic_weight = atomic_weights[1]

    # Coefficients for polynomial.
    c1 = [
        -2.953607e1,
        7.083061e0,
        8.678646e-1,
        -1.221932e0,
        4.052997e-2,
        1.317109e-1,
        -3.265795e-2,
        2.500933e-3,
    ]
    c2 = [
        -2.465188e1,
        4.354679e0,
        -3.553024e0,
        5.573040e0,
        -5.872938e0,
        3.720797e0,
        -1.226919e0,
        1.576657e-1,
    ]

    # Parameters for resonances.
    q = [2.81, 2.51, 2.45, 2.44]
    nu = [1.610, 2.795, 3.817, 4.824]
    gamma = [2.64061e-3, 6.20116e-4, 2.56061e-4, 1.320159e-4]

    # Convert energy in eV into wavelength in angstrom.
    lambd = (const.H * const.C) / (energy * const.EV_TO_ERG) * const.CM_TO_A
    x = np.log10(lambd)
    y = np.zeros(len(lambd))

    mask_1 = lambd < 46.0
    mask_2 = (lambd >= 46.0) & (lambd < 503.97)

    y[mask_1] = sum([c2[i] * (x[mask_1] ** i) for i in range(len(c2))])
    y[mask_2] = sum([c1[i] * (x[mask_2] ** i) for i in range(len(c1))])
    y[mask_2] += sum(
        [
            np.log10(fano_resonance_line(q[i], nu[i], gamma[i], lambd[mask_2]))
            for i in range(len(q))
        ]
    )

    mass_abs_coeff = 10**y * const.AV / He_atomic_weight

    return mass_abs_coeff


def hydrogen(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for hydrogen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)

    x = (
        21.46941
        + (3.0 - 2.060152) * elog
        - 0.1492932 * (elog**2)
        + 5.4634294e-3 * (elog**3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def iron(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for iron where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 707.4
    mask_2 = (energy >= 707.4) & (energy < 7111.2)
    mask_3 = energy >= 7111.2

    x[mask_1] = (
        -15.07332
        + (18.94335 + 3.0) * elog[mask_1]
        - 4.862457 * (elog[mask_1] ** 2)
        + 0.5573765 * (elog[mask_1] ** 3)
        - 3.0065542e-2 * (elog[mask_1] ** 4)
        + 4.9834867e-4 * (elog[mask_1] ** 5)
    )
    x[mask_2] = (
        -253.0979
        + (135.4238 + 3.0) * elog[mask_2]
        - 25.47119 * elog[mask_2] ** 2
        + 2.08867 * (elog[mask_2] ** 3)
        - 6.4264648e-2 * (elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -1.037655
        + (4.022304 + 3.0) * elog[mask_3]
        - 0.3638919 * elog[mask_3] ** 2
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def magnesium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for magnesium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 49.45
    mask_2 = (energy >= 49.45) & (energy < 1303.4)
    mask_3 = energy >= 1303.4

    x[mask_1] = 7.107172 + (0.7359418 + 3.0) * elog[mask_1]
    x[mask_2] = (
        -81.32915
        + (62.2775 + 3.0) * elog[mask_2]
        - 15.00826 * elog[mask_2] ** 2
        + 1.558686 * (elog[mask_2] ** 3)
        - 6.1339621e-2 * (elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -9.161526
        + (10.07448 + 3.0) * elog[mask_3]
        - 1.435878 * elog[mask_3] ** 2
        + 5.2728362e-2 * (elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def neon(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for neon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 867.0
    mask_2 = energy >= 867.0

    x[mask_1] = (
        -3.04041
        + (13.0071 * elog[mask_1])
        + (-1.93205 * (elog[mask_1] ** 2))
        + (0.0977639 * (elog[mask_1] ** 3))
    )
    x[mask_2] = (
        17.6007
        + (3.29278 * elog[mask_2])
        + (-0.263065 * elog[mask_2] ** 2)
        + (5.68290e-3 * elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def nickel(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for nickel where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 853.6
    mask_2 = (energy >= 853.6) & (energy < 8331.6)
    mask_3 = energy >= 8331.6

    x[mask_1] = (
        -7.919931
        + (11.06475 + 3.0) * elog[mask_1]
        - 1.935318 * (elog[mask_1] ** 2)
        + 9.3929626e-2 * (elog[mask_1] ** 3)
    )
    x[mask_2] = (
        3.71129
        + (8.45098 * elog[mask_2])
        + (-0.896656 * elog[mask_2] ** 2)
        + (0.0324889 * elog[mask_2] ** 3)
    )
    x[mask_3] = 28.4989 + (0.485797 * elog[mask_3])

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def nitrogen(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for nitrogen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 401.0
    mask_2 = energy >= 401.0

    x[mask_1] = (
        9.24058
        + (7.02985 * elog[mask_1])
        + (-1.08849 * elog[mask_1] ** 2)
        + (0.0611007 * elog[mask_1] ** 3)
    )
    x[mask_2] = (
        -13.0353
        + (15.4851 * elog[mask_2])
        + (-1.89502 * elog[mask_2] ** 2)
        + (0.0769412 * elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def oxygen(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for oxygen where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 531.7
    mask_2 = energy >= 531.7

    x[mask_1] = (
        2.57264
        + (10.9321 * elog[mask_1])
        + (-1.79383 * elog[mask_1] ** 2)
        + (0.102619 * elog[mask_1] ** 3)
    )
    x[mask_2] = (
        16.53869
        + (0.6428144 + 3.0) * elog[mask_2]
        - 0.3177744 * (elog[mask_2] ** 2)
        + 7.9471897e-3 * (elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def silicon(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for silicon where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 100.6
    mask_2 = (energy >= 100.6) & (energy < 1840.0)
    mask_3 = energy >= 1840.0

    x[mask_1] = (
        -3.066295
        + (7.006248 + 3.0) * elog[mask_1]
        - 0.9627411 * (elog[mask_1] ** 2)
    )
    x[mask_2] = (
        -182.7217
        + (125.061 + 3.0) * elog[mask_2]
        - 29.47269 * (elog[mask_2] ** 2)
        + 3.03284 * (elog[mask_2] ** 3)
        - 0.1173096 * (elog[mask_2] ** 4)
    )
    x[mask_3] = (
        -33.39074
        + (18.42992 + 3.0) * elog[mask_3]
        - 2.385117 * (elog[mask_3] ** 2)
        + 8.887583e-2 * (elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def sodium(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for sodium where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.

    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 1071.7
    mask_2 = energy >= 1071.7

    x[mask_1] = (
        -2737.598
        + (2798.704 + 3.0) * elog[mask_1]
        - 1009.892 * (elog[mask_1] ** 2)
        + 87.16455 * (elog[mask_1] ** 3)
        + 43.20644 * (elog[mask_1] ** 4)
        - 15.27259 * (elog[mask_1] ** 5)
        + 2.180531 * (elog[mask_1] ** 6)
        - 0.1526546 * (elog[mask_1] ** 7)
        + 4.3137977e-3 * (elog[mask_1] ** 8)
    )
    x[mask_2] = (
        1.534019
        + (6.261744 + 3.0) * elog[mask_2]
        - 0.9914126 * (elog[mask_2] ** 2)
        + 3.5278253e-2 * (elog[mask_2] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def sulfur(energy: np.ndarray) -> np.ndarray:
    """
    This function calculates the mass absorption coefficient (mu/rho) for sulfur where mu is the
    linear attenuation coefficient and rho is the density of the material.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.
    Return:
        mass absorption coefficient (np.ndarray): mass absorption coefficient as a function of the energy in cm^2/g.
    """

    elog = np.log(energy)
    x = np.zeros_like(elog)
    mask_1 = energy < 165.0
    mask_2 = (energy >= 165.0) & (energy < 2470.5)
    mask_3 = energy >= 2470.5

    x[mask_1] = (
        598.2911
        + (3.0 - 678.2265) * elog[mask_1]
        + 308.1133 * (elog[mask_1] ** 2)
        - 68.99324 * (elog[mask_1] ** 3)
        + 7.62458 * (elog[mask_1] ** 4)
        - 0.3335031 * (elog[mask_1] ** 5)
    )
    x[mask_2] = (
        3994.831
        + (3.0 - 3693.886) * elog[mask_2]
        + 1417.287 * (elog[mask_2] ** 2)
        - 287.9909 * (elog[mask_2] ** 3)
        + 32.70061 * (elog[mask_2] ** 4)
        - 1.968987 * (elog[mask_2] ** 5)
        + 4.9149349e-2 * (elog[mask_2] ** 6)
    )
    x[mask_3] = (
        -22.49628
        + (14.24599 + 3.0) * elog[mask_3]
        - 1.848444 * (elog[mask_3] ** 2)
        + 6.6506132e-2 * (elog[mask_3] ** 3)
    )

    mass_abs_coeff = np.exp(x) / (energy**3)

    return mass_abs_coeff


def absorption_cross_section_tot(
    energy: np.ndarray, abundances: np.ndarray
) -> np.ndarray:
    """
    This function calculates the effective absorption cross section in units of cm^2/(eV hydrogen atom)
    at energy E in eV for the specified abundances of the elements.
    This function is valid only over the energy range 30 - 10,000 eV.

    Args:
        E (np.ndarray): array of energies of the incoming photons in eV.
        abundances (np.ndarray): array of abundances in log10 relative to hydrogen (= 12.00).

    Return:
        effective cross section (np.ndarray): effective cross section as a function of the energy
        in cm^2/(eV hydrogen atom).
    """

    # Mass absorption cross sections for each element
    mass_abs_coeff = np.zeros((len(atomic_weights), len(energy)))
    mass_abs_coeff[0] = hydrogen(energy)
    mass_abs_coeff[1] = helium(energy)
    mass_abs_coeff[2] = carbon(energy)
    mass_abs_coeff[3] = nitrogen(energy)
    mass_abs_coeff[4] = oxygen(energy)
    mass_abs_coeff[5] = neon(energy)
    mass_abs_coeff[6] = sodium(energy)
    mass_abs_coeff[7] = magnesium(energy)
    mass_abs_coeff[8] = aluminium(energy)
    mass_abs_coeff[9] = silicon(energy)
    mass_abs_coeff[10] = sulfur(energy)
    mass_abs_coeff[11] = chlorine(energy)
    mass_abs_coeff[12] = argon(energy)
    mass_abs_coeff[13] = calcium(energy)
    mass_abs_coeff[14] = chromium(energy)
    mass_abs_coeff[15] = iron(energy)
    mass_abs_coeff[16] = nickel(energy)

    cross_section_tot = np.zeros_like(energy)

    # Calculate the total cross section.
    for i in range(len(atomic_weights)):
        cross_section_tot += (
            atomic_weights[i]
            * mass_abs_coeff[i]
            / const.AV
            * (10 ** (abundances[i] - abundances[0]))
        )

    return cross_section_tot


def absorption_cross_section_approx(energy: np.ndarray) -> np.ndarray:
    """
    This function implements the approximation of Morrison and McCammon (1983)
    to the interstellar photoelectric absorption cross-section. Energy is in eV
    and the resultant cross-section is in cm^2/(eV hydrogen atom). Abundances of other
    elements relative to hydrogen are appropriate for the interstellar medium in
    the solar neighborhood (see Table 1 in Morrison and McCammon 1983). This function
    is valid only over the energy range 30 - 10,000 eV.

    Args:
        energy (np.ndarray): array of energies of the incoming photons in eV.
        abundances (np.ndarray): array of abundances in log10 relative to hydrogen (= 12.00).

    Return:
        effective cross section (np.ndarray): effective cross section as a function of the energy
        in cm^2/(eV hydrogen atom).
    """

    # Energy intervals in keV.
    emax = np.array(
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
    e_keV = energy / 1e3

    # Find the energy interval.
    i = np.searchsorted(emax, e_keV)
    # Handle the case where e_keV exceeds the maximum energy.
    i = np.minimum(i, len(emax) - 1)

    # Calculate cross-section.
    cross_section_ism = (
        (c0[i] + c1[i] * e_keV + c2[i] * e_keV**2) / (e_keV**3) * 1e-24
    )

    return cross_section_ism
