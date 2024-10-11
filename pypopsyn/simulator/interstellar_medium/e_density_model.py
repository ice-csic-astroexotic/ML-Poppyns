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


def compute_tau_sc_327(DM: np.ndarray) -> np.ndarray:
    """
    Given a value of DM compute the scattering timescale.
    We use the empirical fit performed by Krishnakumar et al. (2015), who fitted the scattering times
    obtained at a frequency of 327 MHz (see Section 3, p. 5, right column).

    Args:
        DM (np.ndarray): Dispersion measure in [pc cm^-3].

    Returns:
        (np.ndarray): Values of the scattering timescale in [s].
    """
    # Compute the average tau scattering in [s] at 327 MHz from the empirical formula in Krishnakumar et al. (2015).
    tau_sc_mean = 3.6e-9 * DM**2.2 * (1.0 + 1.94e-3 * DM**2.0)

    # We pick the values of tau_sc from a Gaussian distribution centered on np.log10(tau_sc_mean)
    # with a fiducial sigma of 0.5 in log10 to roughly reproduce the scatter in the data as in Fig. 3 in
    # Krishnakumar et al. (2015).
    tau_sc = 10 ** np.random.normal(np.log10(tau_sc_mean), 0.5)

    return tau_sc


def compute_tau_sc_f(tau_sc: np.ndarray, f: float) -> np.ndarray:
    """
    Rescaling the scattering timescale at the given specified observation frequency.
    To rescale to any frequency we assume a Kolmogorov spectrum tau(f) ~ f^-4.4.

    Args:
        tau_sc (np.ndarray): Scattering timescale at 327 MHz in [s].
        f (np.ndarray): Frequency at which the scattering timescale is computed [Hz].

    Returns:
        (np.ndarray): Values of the scattering timescale at the frequency nu in [s].
    """
    tau_sc_f = tau_sc * (f / 327.0e6) ** (-4.4)

    return tau_sc_f
