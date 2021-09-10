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


class SurveyRadioBase:
    """
    Base class for any radio survey model to ensure that a common interface between all of
    them is respected. The following abstract methods must be implemented or an error
    will be raised.
    """

    def __init__(
        self,
        beta: float,
        G0: float,
        t_obs: float,
        t_samp: float,
        T_sys: float,
        nu_central: float,
        BW: float,
        channel_width: float,
        n_p: float,
        FWHM: float,
        SN_th: float,
        RA_range: np.ndarray,
        DEC_range: np.ndarray,
        l_range: np.ndarray,
        b_range: np.ndarray,
    ):
        """
        Radio survey initialization.

        Args:
            beta (float): degradation factor.
            G0 (float): gain at the beam center [K Jy^(-1)].
            t_obs (float): integration time [s].
            t_samp (float): sampling time [s].
            T_sys (float): system temperature [K].
            nu_central (float): central frequency of the bandwidth [Hz].
            BW (float): frequency bandwidth [Hz].
            channel_width (float): width of a single frequency channel [Hz].
            n_p (float): number of polarizations.
            FWHM (float): FWHM of the beam [arcmin].
            SN_th (float): threshold signal to noise ratio.
            RA_range (np.ndarray): range of the sky covered by the survey in RA [deg].
            DEC_range (np.ndarray): range of the sky covered by the survey in DEC [deg].
            l_range (np.ndarray): range of the sky covered by the survey in galactic longitude l [deg].
            b_range (np.ndarray): range of the sky covered by the survey in galactic latitude b [deg].

        Returns:
            Nothing.

        """

        self.beta = beta
        self.G0 = G0
        self.t_obs = t_obs
        self.t_samp = t_samp
        self.T_sys = T_sys
        self.nu_central = nu_central
        self.BW = BW
        self.channel_width = channel_width
        self.n_p = n_p
        self.FWHM = FWHM
        self.SN_th = SN_th
        self.RA_range = RA_range
        self.DEC_range = DEC_range
        self.l_range = l_range
        self.b_range = b_range

    def sky_coverage(
        self,
        RA: np.ndarray,
        DEC: np.ndarray,
        l_gal: np.ndarray,
        b_gal: np.ndarray,
    ) -> np.ndarray:
        """
            Determine which neutron stars are in the sky region covered by the survey.

            Args:
                RA (np.ndarray): right ascension in [deg] defined between [0, 360] deg in ICRS frame.
                DEC (np.ndarray): declination in [deg] defined between [-90, 90] deg in ICRS frame.
                l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
                b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.

            Returns:
                (np.ndarray): array of boolean variables: true if the pulsar is in the covered sky region, false if not.
        """
        coverage = (
            (RA > self.RA_range[0])
            & (RA < self.RA_range[1])
            & (DEC > self.DEC_range[0])
            & (DEC < self.DEC_range[1])
            & (l_gal > self.l_range[0])
            & (l_gal < self.l_range[1])
            & (b_gal > self.b_range[0])
            & (b_gal < self.b_range[1])
        )

        return coverage

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
            l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.
            w_int (np.ndarray): intrinsic pulse width in [s].
            P (np.ndarray): spin period in [s].

        Returns:
            (np.ndarray): array of boolean variables: true if the pulsar is detected, false if not.
        """
        # Store the total number of sources.
        n = len(S_radio)

        # Compute the effective pulse width.
        w_eff = effective_pulse_width(
            w_int, DM, self.channel_width, self.nu_central, self.t_samp,
        )
        # Draw a random offset from the telescope beam center.
        offset2 = self.detection_offset(n)
        # Compute the gain corresponding to the offset detections.
        G = self.gain_gaussian_beam(offset2)

        # Compute the Sky temperature in the coordinates of each detection at the central frequency of the survey.
        T_sky = sky_temperature(l_gal, b_gal, self.nu_central)

        SN_detection = np.zeros(n)

        SN_detection = self.antenna_equation(S_radio, G, w_eff, P, T_sky)

        detected = SN_detection > self.SN_th

        return detected


class SurveyRadioPMPS(SurveyRadioBase):
    """
    Class that model the Parks Multibeam Pulsar Survey.
    The survey parameters are taken from Manchester et al. (2001) (see also Bates et al. 2014 and Chakraborty et al. 2020).
    """

    def __init__(
        self,
        beta=1.2,
        G0=0.7,
        t_obs=2100.0,
        t_samp=250.0e-6,
        T_sys=25.0,
        nu_central=1.352e9,
        BW=288.0e6,
        channel_width=3.0e6,
        n_p=2,
        FWHM=14.0,
        SN_th=9.0,
        RA_range=np.array([0.0, 360.0]),
        DEC_range=np.array([-90.0, 90.0]),
        l_range=np.array([-150.0, 50.0]),
        b_range=np.array([-6.0, 6.0]),
    ) -> None:

        super().__init__(
            beta,
            G0,
            t_obs,
            t_samp,
            T_sys,
            nu_central,
            BW,
            channel_width,
            n_p,
            FWHM,
            SN_th,
            RA_range,
            DEC_range,
            l_range,
            b_range,
        )


class SurveyRadioSMPS(SurveyRadioBase):
    """
    Class that model the Swinburne Multibeam Pulsar Survey.
    # The survey parameters are taken from Edwards et al. (2001) (see also Chakraborty et al. 2020).
    """

    def __init__(
        self,
        beta=1.5,
        G0=0.64,
        t_obs=265.0,
        t_samp=125.0e-6,
        T_sys=25.0,
        nu_central=1.374e9,
        BW=288.0e6,
        channel_width=3.0e6,
        n_p=2,
        FWHM=14.0,
        SN_th=9.0,
        RA_range=np.array([0.0, 360.0]),
        DEC_range=np.array([-90.0, 90.0]),
        l_range=np.array([-100.0, 50.0]),
        b_range=np.array([-5.0, 15.0]),
    ) -> None:
        super().__init__(
            beta,
            G0,
            t_obs,
            t_samp,
            T_sys,
            nu_central,
            BW,
            channel_width,
            n_p,
            FWHM,
            SN_th,
            RA_range,
            DEC_range,
            l_range,
            b_range,
        )
