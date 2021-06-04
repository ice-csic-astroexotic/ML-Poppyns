"""
Model for the free electron density to compute the DM values and the scattering timescales.

We use the library pygedm avalilable from astropy.
See https://pygedm.readthedocs.io/en/latest/pygedm.html for the related documentation.

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

from typing import Tuple

import numpy as np
import pygedm


def compute_DM(
    l_gal: np.ndarray, b_gal: np.ndarray, d: np.ndarray, ed_model: str,
) -> np.ndarray:
    """
    Given a specified electron density model between 'ymw16' and 'ne2001' compute
    the dispersion measure DM related to a given heliocentric distance.

    Args:
        l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.
        d (np.ndarray): heliocentric distance in [kpc].
        ed_model (str): free electron density model, either 'ymw16' or 'ne2001'.
        NS_number (int): number of neutron stars.

    Returns:
        (np.ndarray): values of the DM in [pc cm^-3].
    """
    # Store the number of object that need computation of DM.
    n = len(l_gal)

    DM = np.zeros(n)

    d_pc = d * 1000  # Convert distance from kpc to pc.

    for i in range(n):
        # The function dist_to_dm accept only floats as input and the distance must be in [pc].
        dm, _ = pygedm.dist_to_dm(l_gal[i], b_gal[i], d_pc[i], method=ed_model)

        DM[i] = dm.value

    return DM


def compute_tau_sc(DM: np.ndarray, nu: float) -> np.ndarray:
    """
    Given a value of DM compute the scattering timescale at the given specified observation frequency.
    We use the empirical fit performed by Krishnakumar et al. (2015) who fitted the scattering times
    obtained at a frequency of 327 MHz. To rescale to any frequency we assume a Kolmogorov spectrum
    tau(nu) ~ nu^-4.4.

    Args:
        DM (np.ndarray): dispersion measure in [pc cm^-3].
        nu (np.ndarray): frequency at which the observation is computed [Hz].

    Returns:
        (np.ndarray): values of the scattering timescale at the frequency nu in [s].
    """
    # Compute the average tau scattering in [s] at 327 MHz from the empirical formula in Krishnakumar et al. (2015).
    tau_sc_mean = 3.6e-9 * DM ** 2.2 * (1.0 + 1.94e-3 * DM ** 2.0)

    # We pick the values of tau_sc from a gaussian distribution centered on np.log10(tau_sc_mean)
    # with a fiducial sigma of 0.5 in log10 to roughly reproduce the scatter in the data as in Fig. 3 in
    # krishnakumar et al. (2015).
    tau_sc = 10 ** np.random.normal(np.log10(tau_sc_mean), 0.5)

    # Convert the scattering time to a given observation frequency nu assuming a Kolmogorov spectrum.
    tau_sc_nu = tau_sc * (nu / 327.0e6) ** (-4.4)

    return tau_sc_nu
