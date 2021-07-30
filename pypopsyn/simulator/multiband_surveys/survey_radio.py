"""
Model for the pulsar radio surveys.

We consider the following surveys:

1) PMPS: the Parks Multibeam Pulsar Survey (see Manchester et al. 2001)

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

import abc
from typing import Tuple

import numpy as np

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.interstellar_medium.e_density_model as edm


def smearing_in_channel(
    DM: np.ndarray, channel_width: float, nu: float
) -> np.ndarray:
    """
    Dispersive smearing inside a single frequency channel in [s] evaluated for
    the central frequency of the survey. See eq. (27) in Bates et al. (2014)
    and appendix A2.4 of Handbook of pulsar astronomy by Lorimer and Kramer (2004).

    Args:
        DM (np.ndarray): dispersion measure in [pc cm^-3].
        channel_width (float): width in frequency of a single frequency channel of the receiver in [Hz].
        nu (float): central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): intra-channel dispersive smearing in [s].
    """

    dt = (
        2
        * const.e ** 2
        / (2 * np.pi * const.m_e * const.c)
        * channel_width
        / nu ** 3.0
        * DM
        * const.PC_TO_CM
    )

    return dt


def effective_pulse_width(
    w_int: np.ndarray,
    DM: np.ndarray,
    channel_width: float,
    nu: float,
    t_samp: float,
) -> np.ndarray:
    """
    Measured effective pulse width which is smeared out by the inter-channel dispersion, the scattering
    with the interstellar medium and by the instrumental sampling time.

    Args:
        w_int (np.ndarray): intrinsic pulse width in [s].
        DM (np.ndarray): dispersion measure in [pc cm^-3].
        channel_width (float): width in frequency of a single frequency channel of the receiver in [Hz].
        nu (float): central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): measured effective pulse width in [s].
    """

    tau_DM = smearing_in_channel(DM, channel_width, nu)
    tau_sc = edm.compute_tau_sc(DM, nu)
    w_eff = np.sqrt(w_int ** 2 + tau_sc ** 2 + tau_DM ** 2 + t_samp ** 2)

    return w_eff


def sky_temperature(
    l_gal: np.ndarray, b_gal: np.ndarray, nu: float
) -> np.ndarray:
    """
    Sky temperature as a function of galactic longitude and latitude (l, b) and frequency.
    We use an empirical fit from Narayan (1987) and rescale to the given frequency using
    a relation from Johnston et al. 1992 (see also Yousifov & Kucuk 2004).

    Args:
        l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.
        nu (np.ndarray): central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): measured sky temperature in [K] as a function of the galactic coordinates at frequency nu.
    """

    # Sky temperature at 408 Mhz from Narayan (1987).
    T_sky_400 = 25.0 + 275.0 / (
        (1.0 + (l_gal / 42.0) ** 2) * (1.0 + (b_gal / 3.0) ** 2)
    )

    # Rescale to the wanted frequency.
    T_sky_nu = T_sky_400 * (408.0e6 / nu) ** 2.6

    return T_sky_nu


class SurveyRadioPMPS:
    """
    Class that model the Parks Multibeam Pulsar Survey.
    """

    def __init__(self) -> None:

        # The survey parameters are taken from Manchester et al. (2001) and Bates et al. (2014).
        self.beta = 1.2  # degradation factor.
        self.G0 = 0.7  # gain at the beam center [K Jy^(-1)].
        self.t_obs = 2100.0  # integration time [s].
        self.t_samp = 250.0e-6  # sampling time [s].
        self.T_sys = 25.0  # system temperature [K].
        self.nu_central = 1.352e9  # central frequency of the bandwidth [Hz]
        self.BW = 288.0e6  # frequency bandwidth [Hz].
        self.channel_width = 3.0e6  # width of a single frequency channel [Hz].
        self.n_p = 2  # number of polarizations.
        self.FWHM = 14.0  # FWHM of the beam [arcmin].
        self.SN_th = 9.0  # threshold signal to noise ratio.
        self.RA_min = (
            0.0  # range of the sky visible by the survey in RA [deg].
        )
        self.RA_max = 360.0
        self.DEC_min = (
            -90.0
        )  # range of the sky visible by the survey in DEC [deg].
        self.DEC_max = +90.0
        self.l_min = (
            -150.0
        )  # range of the sky visible by the survey in galactic longitude l [deg].
        self.l_max = 50.0
        self.b_min = (
            -6.0
        )  # range of the sky visible by the survey in galactic latitude b [deg].
        self.b_max = +6.0

    def detection_offset(self, n_detection: int) -> np.ndarray:
        """
        Generating a random offset with respect to the beam center for the detections.
        A gaussian beam pattern is assumed (see Lorimer et al. 1993).

        Args:
            n_detection (int): number of detection to simulate.

        Returns:
            (np.ndarray): square of the offset from the beam center for each detection in [arcmin^2] .
        """
        offset2 = np.random.uniform(0.0, self.FWHM ** 2 / 4.0, n_detection)

        return offset2

    def gain_gaussian_beam(self, offset2: np.ndarray) -> np.ndarray:
        """
        This method simulate the gain pattern of a receiver.
        A gaussian beam pattern is assumed (see Lorimer et al. 1993).

        Args:
            offset2 (np.ndarray): squared offset from the beam center in [arcmin^2].

        Returns:
            (np.ndarray): gain of the telescope for the given offset in [K Jy^(-1)].
        """

        G = self.G0 * np.exp(-2.77 * offset2 / self.FWHM ** 2)

        return G

    def antenna_equation(
        self,
        S_radio: np.ndarray,
        G: np.ndarray,
        w_eff: np.ndarray,
        P: np.ndarray,
        T_sky: np.ndarray,
    ) -> np.ndarray:
        """
        Antenna equation used to compute the signal to noise ratio of each pulsars given the radio flux
        at a given frequency nu, the effective pulse width, the spin period and the survey parameters.

        Args:
            S_radio (np.ndarray): radio flux in Jy.
            G (np.ndarray): Gain of the telescope for the given detection in [K Jy^(-1)].
            w_eff (np.ndarray): effective pulse width in [s].
            P (np.ndarray): spin period in [s].
            T_sky (np.ndarray): Sky temperature for every detection in [K].

        Returns:
            (np.ndarray): signal to noise ratio of the detection.
        """
        SN = np.zeros(len(S_radio))

        # If the effective pulse width is larger than the spin period, then the pulsar is not detected.
        cond = w_eff < P

        # Compute the SN of each detection using the antenna equation.
        SN[cond] = (
            S_radio[cond]
            * G[cond]
            * np.sqrt(self.n_p * self.t_obs * self.BW)
            * np.sqrt((P[cond] - w_eff[cond]) / w_eff[cond])
            / (self.beta * (self.T_sys + T_sky[cond]))
        )

        return SN

    def detect(
        self,
        S_radio: np.ndarray,
        DM: np.ndarray,
        RA: np.ndarray,
        DEC: np.ndarray,
        l_gal: np.ndarray,
        b_gal: np.ndarray,
        w_int: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate a detection: if the measured SN surpasses the threshold SN_th of the survey
        then the pulsar is detected.

        Args:
            S_radio (np.ndarray): radio flux in Jy.
            DM (np.ndarray): dispersion measure in [pc cm^-3].
            RA (np.ndarray): right ascension in [deg] defined between [0, 360] deg in ICRS frame.
            DEC (np.ndarray): declination in [deg] defined between [-90, 90] deg in ICRS frame.
            l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.
            w_int (np.ndarray): intrinsic pulse width in [s].
            P (np.ndarray): spin period in [s].

        Returns:
            (np.ndarray): array of boolean variables: true if the pulsar is detected, false if not.
        """
        # Store the total number of sources.
        n = len(S_radio)

        visibility = (
            (RA > self.RA_min)
            & (RA < self.RA_max)
            & (DEC > self.DEC_min)
            & (DEC < self.DEC_max)
            & (l_gal > self.l_min)
            & (l_gal < self.l_max)
            & (b_gal > self.b_min)
            & (b_gal < self.b_max)
        )

        # Store the number of potentially detectable sources.
        n_vis = len(S_radio[visibility])

        # Compute the effective pulse width.
        w_eff = effective_pulse_width(
            w_int[visibility],
            DM[visibility],
            self.channel_width,
            self.nu_central,
            self.t_samp,
        )
        # Draw a random offset from the telescope beam center.
        offset2 = self.detection_offset(n_vis)
        # Compute the gain corresponding to the offset detections.
        G = self.gain_gaussian_beam(offset2)

        # Compute the Sky temperature in the coordinates of each detection at the central frequency of the survey.
        T_sky = sky_temperature(
            l_gal[visibility], b_gal[visibility], self.nu_central
        )

        SN_detection = np.zeros(n)

        SN_detection[visibility] = self.antenna_equation(
            S_radio[visibility], G, w_eff, P[visibility], T_sky
        )

        detected = SN_detection > self.SN_th

        return detected
