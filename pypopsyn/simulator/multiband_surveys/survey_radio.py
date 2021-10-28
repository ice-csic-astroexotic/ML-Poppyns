"""
Model for the pulsar radio surveys.

We consider the following surveys:

1) PMPS: the Parks Multibeam Pulsar Survey (see Manchester et al. 2001, Lorimer et al. 2006)
2) SMPS: the Swinburne Multibeam Pulsar Survey (see Edwards et al. 2001, Jacoby et al. 2009)

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

import json

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

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
    with the interstellar medium and by the instrumental sampling time (see eq. 2 in Cordes & McLaughlin 2003).

    Args:
        w_int (np.ndarray): intrinsic pulse width in [s].
        DM (np.ndarray): dispersion measure in [pc cm^-3].
        channel_width (float): width in frequency of a single frequency channel of the receiver in [Hz].
        nu (float): central frequency at which the observation is performed [Hz].
        t_samp (float): sampling time for the radio survey [s].

    Returns:
        (np.ndarray): measured effective pulse width in [s].
    """

    tau_DM = smearing_in_channel(DM, channel_width, nu)
    tau_sc = edm.compute_tau_sc(DM, nu)
    w_eff = np.sqrt(w_int ** 2 + tau_sc ** 2 + tau_DM ** 2 + t_samp ** 2)

    return w_eff


def sky_temperature_approx(
    l_gal: np.ndarray, b_gal: np.ndarray, nu: float
) -> np.ndarray:
    """
    Sky temperature as a function of galactic longitude and latitude (l, b) and frequency.
    We use an empirical fit from Narayan (1987) and rescale to the given frequency using
    a relation from Johnston et al. (1992) (see also eq. 5 in Yusifov & Küçük 2004).

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

    # Rescale to the wanted frequency assuming a sky temperature spectral index of -2.6 (see Lawson et al. 1987,
    # Johnston et al. 1992).
    T_sky_nu = T_sky_400 * (408.0e6 / nu) ** 2.6

    return T_sky_nu


def sky_temperature(
    l_gal: np.ndarray, b_gal: np.ndarray, nu: float
) -> np.ndarray:
    """
    Sky temperature as a function of galactic longitude and latitude (l, b) and frequency.
    We use the map from Haslam et al. (1981), downloadable here:
    https://lambda.gsfc.nasa.gov/product/foreground/haslam_408.cfm.

    Args:
        l_gal (np.ndarray): galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): galactic latitude in [deg] defined between [-90, 90] deg.
        nu (np.ndarray): central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): measured sky temperature in [K] as a function of the galactic coordinates at frequency nu.
    """

    # Read the sky temperature map.
    file = "pypopsyn/simulator/multiband_surveys/Tsky_map_haslam81.fits"
    hdulist = fits.open(file)
    hdu = hdulist["TEMPERATURE"]
    data = hdu.data

    # Convert coordinates into astropy coordinates object.
    coord = SkyCoord(l_gal, b_gal, frame="galactic", unit="deg")

    # Convert sky coordinates into pixel coordinates and extract the temperatures.
    wcs = WCS(hdu.header)
    x_pixel, y_pixel = wcs.world_to_pixel(coord)
    x_pixel = x_pixel.astype(int)
    y_pixel = y_pixel.astype(int)
    T_sky_400 = data[y_pixel, x_pixel]

    # Rescale to the wanted frequency assuming a sky temperature spectral index of -2.6 (see Lawson et al. 1987,
    # Johnston et al. 1992).
    T_sky_nu = T_sky_400 * (408.0e6 / nu) ** 2.6

    return T_sky_nu


class SurveyRadio:
    """
    Class for any radio survey with a gaussian telescope beam pattern.
    The parameters for the survey are imported from a JSON file.
    """

    def __import_parameters(self, parameters_path):
        """
        Import dataset statistics for normalization and standardization.

        This routine import the parameters of a radio survey.

        Args:
            parameters_path (str): path to the survey_parameter.json file containing the parameters
                of the radio survey.

        Returns:
            Nothing.

        """

        # Load parameters from JSON file.
        with open(parameters_path) as read_file:
            self.parameters = json.load(read_file)

        # Save the parameters.
        self.deg_factor = self.parameters["deg_factor"]
        self.G0 = self.parameters["G0"]
        self.t_obs = self.parameters["t_obs"]
        self.t_samp = self.parameters["t_samp"]
        self.T_sys = self.parameters["T_sys"]
        self.nu_central = self.parameters["nu_central"]
        self.BW = self.parameters["BW"]
        self.channel_width = self.parameters["channel_width"]
        self.n_pol = self.parameters["n_pol"]
        self.FWHM = self.parameters["FWHM"]
        self.SN_th = self.parameters["SN_th"]
        self.RA_range = self.parameters["RA_range"]
        self.DEC_range = self.parameters["DEC_range"]
        self.l_range = self.parameters["l_range"]
        self.b_range_abs = self.parameters["b_range_abs"]

    def __init__(
        self, parameters_path,
    ):
        """
        Radio survey initialization.

        Args:
            parameters_path (str): path to the survey_parameter.json file containing the parameters
                of the radio survey.

        Returns:
            Nothing.

        """

        self.__import_parameters(parameters_path)

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
            & (np.abs(b_gal) > self.b_range_abs[0])
            & (np.abs(b_gal) < self.b_range_abs[1])
        )

        return coverage

    def detection_offset(self, n_detection: int) -> np.ndarray:
        """
        Generating a random offset with respect to the beam center for the detections.
        A Gaussian beam pattern is assumed (see Lorimer et al. 1993).

        Args:
            n_detection (int): number of detections to simulate.

        Returns:
            (np.ndarray): square of the offset from the beam center for each detection in [arcmin^2] .
        """
        offset2 = np.random.uniform(0.0, self.FWHM ** 2 / 4.0, n_detection)

        return offset2

    def gain_gaussian_beam(self, offset2: np.ndarray) -> np.ndarray:
        """
        This method simulates the gain pattern of a receiver.
        A Gaussian beam pattern is assumed (see eq. 14 in Lorimer et al. 1993 and eq. 29 in Bates et. al 2014).

        Args:
            offset2 (np.ndarray): squared offset from the beam center in [arcmin^2].

        Returns:
            (np.ndarray): gain of the telescope for the given offset in [K Jy^(-1)].
        """

        G = self.G0 * np.exp(-2.77 * offset2 / self.FWHM ** 2)

        return G

    def radiometer_equation(
        self,
        S_radio: np.ndarray,
        G: np.ndarray,
        w_eff: np.ndarray,
        P: np.ndarray,
        T_sky: np.ndarray,
    ) -> np.ndarray:
        """
        Antenna equation used to compute the signal to noise ratio of each pulsars given the radio flux
        at a given frequency nu, the effective pulse width, the spin period and the survey parameters
        (see eq. A1.22 in Lorimer & Kramer 2005).

        Args:
            S_radio (np.ndarray): radio flux in [Jy].
            G (np.ndarray): gain of the telescope for the given detection in [K Jy^(-1)].
            w_eff (np.ndarray): effective pulse width in [s].
            P (np.ndarray): spin period in [s].
            T_sky (np.ndarray): sky temperature for every detection in [K].

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
            * np.sqrt(self.n_pol * self.t_obs * self.BW)
            * np.sqrt((P[cond] - w_eff[cond]) / w_eff[cond])
            / (self.deg_factor * (self.T_sys + T_sky[cond]))
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
            S_radio (np.ndarray): total radio flux from a source in Jy.
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

        SN_detection = self.radiometer_equation(S_radio, G, w_eff, P, T_sky)

        detected = SN_detection > self.SN_th

        return detected
