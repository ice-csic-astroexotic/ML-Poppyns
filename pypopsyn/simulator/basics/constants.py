"""
    Constants module.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

# Unit conversions.

KPC_TO_KM = 3.08567758e16  # Convert from [kpc] to [km].
KPC_TO_CM = 3.08567758e21  # Convert from [kpc] to [cm].
PC_TO_CM = 3.08567758e18  # Convert from [pc] to [cm].
KM_TO_CM = 100000.0  # Convert from [km] to [cm].
YR_TO_S = 3600.0 * 24 * 365  # Convert from [yr] to [s].
MILLIJY_TO_ERG = 1.0e-26  # Convert [mJy] to [erg cm^-2 s^-1 Hz^-1]
JY_TO_ERG = 1.0e-23  # Convert [Jy] to [erg cm^-2 s^-1 Hz^-1]
EV_TO_ERG = 1.60218e-12  # Convert [eV] to [erg].
RYD_TO_ERG = 2.1798741e-11  # Convert [Rydberg] to [erg].
CM_TO_A = 1.0e8  # Convert [cm] to [Angstrom].

# Physical constants.

M_SUN = 2.0e33  # Sun's mass in [g].
M_E = 9.10938356e-28  # Electron's mass in [g].
C = 29979245800.0  # Speed of light [cm/s].
E = 4.80320425e-10  # Electric charge in [statC] = [cm^(3/2)g^(1/2)/s].
G = 6.67e-8  # Gravitational constant in [cm^3 g^-1 s^-2].
H = 6.6261e-27  # Planck constant in [erg s].
AV = 6.022045e23  # Avogadro's number [mol^-1].
K_B = 1.380658e-16  # Boltzmann constant [erg K^-1].
SIGMA_SB = 5.67051e-5  # Stefan-Boltzmann constant [erg cm^-2 K^-4 s^-1].

G_KPC_YR = (
    G / (KPC_TO_CM**3) * YR_TO_S**2
)  # Gravitational constant in [kpc^3 g^-1 yr^-2].
