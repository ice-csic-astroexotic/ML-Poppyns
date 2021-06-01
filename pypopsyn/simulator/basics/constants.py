"""
Constants module.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
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

# Unit conversions.

KPC_TO_KM = 3.08567758e16  # Convert from [kpc] to [km].
KPC_TO_CM = 3.08567758e21  # Convert from [kpc] to [cm].
KM_TO_CM = 100000.0  # Convert from [km] to [cm].
YR_TO_S = 3600.0 * 24 * 365  # Convert from [yr] to [s].
MJY_TO_ERG = 1.0e-26  # Convert [mJy] to [erg cm^-2 s^-1 Hz^-1]
JY_TO_ERG = 1.0e-23  # Convert [Jy] to [erg cm^-2 s^-1 Hz^-1]

# Physical constants.

M_SUN = 2.0e33  # Sun's mass in [g].
c = 29979245800.0  # Speed of light [cm/s]
e = 4.80320425e-10  # Electric charge in [statC] = [cm^(3/2)g^(1/2)/s]
G = 6.67e-8  # Gravitational constant in [cm^3 g^-1 s^-2].

G_KPC_YR = (
    G / (KPC_TO_CM ** 3) * YR_TO_S ** 2
)  # Gravitational constant in [kpc^3 g^-1 yr^-2].
