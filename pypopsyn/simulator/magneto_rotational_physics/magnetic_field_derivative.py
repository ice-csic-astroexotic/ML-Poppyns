"""
Time derivative of the pulsar dipolar magnetic field.

Authors:

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

import numpy as np
from numba import float64, jit

import pypopsyn.simulator.basics.constants as const
from pypopsyn.simulator.configuration import cfg

# Redefining global variables to allow type specification.
# Necessary right now in order to get JIT to work.
L_cfg: float = cfg["L"]
sigma_cfg: float = cfg["sigma"]
n_e_cfg: float = cfg["n_e"]


@jit(float64(float64, float64))
def timescale_ohmic(L: float, sigma: float) -> float:
    """
    Calculating the ohmic diffusion timescale for a given conductivity and characteristic magnetic
    field length scale. Note that for our purposes, we neglect the fact that both quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the ohmic diffusion process. For our choices see the configuration file.

    Args:
        L (float): characteristic length scale on which the magnetic field varies, measured in [cm].
        sigma (float): conductivity of the dominating dissipative process, measure in [1/s].

    Returns:
        (float): ohmic diffusion timescale in [yr].
    """

    tau_ohm = 4 * np.pi * sigma * L ** 2 / (const.C ** 2 * const.YR_TO_S)

    return tau_ohm


@jit(float64(float64, float64, float64))
def timescale_Hall(B: float, L: float, n_e: float) -> float:
    """
    Calculating the Hall timescale for a given field strength, characteristic magnetic field length
    scale and electron density. Note that for our purposes, we neglect the fact that all quantities can
    vary significantly with depth inside the neutron star, and we simply use effective quantities
    that reflect the conservative Hall process. For our choices of L and n_e see the configuration file.
    B will be identified with the initial dipolar magnetic field components at the pulsars' pole.

    Args:
        B (float): (local) magnetic field strength, measured in [G].
        L (float): characteristic length scale on which the magnetic field varies, measured in [cm].
        n_e (float): electron density, measured in [g/cm^3].

    Returns:
        (float): Hall timescale in [yr].
    """

    tau_Hall = (
        4 * np.pi * const.E * n_e * L ** 2 / (const.C * B * const.YR_TO_S)
    )

    return tau_Hall


@jit(float64(float64, float64))
def field_derivative(B: float, B_initial: float) -> float:
    """
    Calculating the change in the magnetic field strength of a pulsar based on a simplified
    differential equation (see eq. (18) of Aguilera et al. (2008)) that captures the
    characteristics of more complication numerical simulations of pulsar magnetic field
    evolution, i.e., at early timescales the Hall evolution dominates, while at late times
    the exponential magnetic field decay due to Ohmic dissipation kicks in. Note that as
    explained in Aguilera et al. (2008) the Hall timescale corresponds to that of the initial
    field strength.

    Args:
        B (float): pulsar's magnetic field magnitudes evolving with time, measured in [G].
        B_initial (float): pulsar's initial magnetic field magnitudes, measured in [G].

    Returns:
        (np.ndarray): magnetic field derivatives for a simulated pulsars in [G/yr].
    """

    tau_ohm = timescale_ohmic(L_cfg, sigma_cfg)
    tau_Hall = timescale_Hall(B_initial, L_cfg, n_e_cfg)

    B_deriv = -B / tau_ohm - B ** 2 / (tau_Hall * B_initial)

    return B_deriv
