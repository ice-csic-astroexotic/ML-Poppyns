""" Constants module.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

# Unit conversions.

KPC_TO_KM = 3.08567758e16  # Convert from [kpc] to [km].
KPC_TO_CM = 3.08567758e21  # Convert from [kpc] to [cm].
KM_TO_CM = 100000  # Convert from [km] to [cm].
YR_TO_S = 3600 * 24 * 365  # Convert from [yr] to [s].

# Physical constants.

M_SUN = 2.0e33  # Sun mass [g].

G = 6.67e-8  # Gravitational constant [cm^3 g^-1 s^-2].

G_KPC_YR = (
    G / (KPC_TO_CM ** 3) * YR_TO_S ** 2
)  # Gravitational constant [kpc^3 g^-1 yr^-2].
