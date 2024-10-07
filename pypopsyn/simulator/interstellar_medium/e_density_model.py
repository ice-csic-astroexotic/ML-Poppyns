"""
    Model for the free electron density to compute the DM values and the scattering timescales.

    We use the library pygedm available from astropy (see also Price et al. 2021)
    See https://pygedm.readthedocs.io/en/latest/pygedm.html for the related documentation.

    Authors:

         Michele Ronchi (ronchi@ice.csic.es)
"""

import numpy as np
import pygedm


def compute_DM(
    l_gal: np.ndarray,
    b_gal: np.ndarray,
    d: np.ndarray,
    ed_model: str,
) -> np.ndarray:
    """
    Given a specified electron density model (either 'ymw16' or 'ne2001') compute
    the dispersion measures DMs related to the given heliocentric distances.

    Args:
        l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
        d (np.ndarray): Heliocentric distance in [kpc].
        ed_model (str): Free electron density model, either 'ymw16' or 'ne2001'.

    Returns:
        (np.ndarray): Values of the DM in [pc cm^-3].
    """
    # Store the number of objects that need computation of DM.
    n = len(l_gal)

    DM = np.zeros(n)

    # Convert distance from [kpc] to [pc].
    d_pc = d * 1000

    for i in range(n):
        # The function dist_to_dm accepts only floats as input and the distance must be in [pc].
        dm, _ = pygedm.dist_to_dm(l_gal[i], b_gal[i], d_pc[i], method=ed_model)

        DM[i] = dm.value

    return DM


def compute_tau_sc(tau_sc: np.ndarray, f: float) -> np.ndarray:
    """
    Rescale the value of tau_sc to any frequency we assume a Kolmogorov spectrum tau(f) ~ f^-4.4.

    Args:
        tau_sc (np.ndarray): Scattering timescales in [s].
        f (np.ndarray): Frequency at which the scattering timescale is computed [Hz].

    Returns:
        (np.ndarray): Values of the scattering timescale at the frequency f in [s].
    """

    # Convert the scattering time to a given observation frequency f assuming a Kolmogorov spectrum.
    tau_sc_f = tau_sc * (f / 327.0e6) ** (-4.4)

    return tau_sc_f
