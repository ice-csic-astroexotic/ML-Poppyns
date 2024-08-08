"""
    Model for the pulsar radio surveys.

    We consider the following surveys:

    1. PMPS: the Parks Multibeam Pulsar Survey (see Manchester et al. 2001, Lorimer et al. 2006)
    2. SMPS: the Swinburne Parkes Multibeam Pulsar Survey (see Edwards et al. 2001, Jacoby et al. 2009)
    3. HTRU: the High Time Resolution Universe Survey (see Keith et al. 2018)

    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
            Celsa Pardo Araujo (pardo@ice.csic.es)
"""
import json
import pathlib
from typing import Tuple

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.multiband_emission.emission_radio as er
from pypopsyn.simulator.config_simulator import cfg


def smearing_in_channel(
    DM: np.ndarray, channel_width: float, nu: float
) -> np.ndarray:
    """
    Dispersive smearing inside a single frequency channel in [s] evaluated for
    the central frequency of the survey. See eq. (27) in Bates et al. (2014)
    and appendix A2.4 of Handbook of pulsar astronomy by Lorimer and Kramer (2004).

    Args:
        DM (np.ndarray): Dispersion measure in [pc cm^-3].
        channel_width (float): Width in frequency of a single frequency channel of the receiver in [Hz].
        nu (float): Central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): Intra-channel dispersive smearing in [s].
    """

    dt = (
        2
        * const.E**2
        / (2 * np.pi * const.M_E * const.C)
        * channel_width
        / nu**3.0
        * DM
        * const.PC_TO_CM
    )

    return dt


def effective_pulse_width(
    w_int: np.ndarray,
    DM: np.ndarray,
    channel_width: float,
    f: float,
    t_samp: float,
) -> np.ndarray:
    """
    Measured effective pulse width which is smeared out by the inter-channel dispersion, the scattering
    with the interstellar medium and by the instrumental sampling time (see eq. (2) in Cordes & McLaughlin 2003).

    Args:
        w_int (np.ndarray): Intrinsic pulse width in [s].
        DM (np.ndarray): Dispersion measure in [pc cm^-3].
        channel_width (float): Width in frequency of a single frequency channel of the receiver in [Hz].
        f (float): Central frequency at which the observation is performed [Hz].
        t_samp (float): Sampling time for the radio survey [s].

    Returns:
        (np.ndarray): Measured effective pulse width in [s].
    """

    tau_DM = smearing_in_channel(DM, channel_width, f)
    tau_sc = edm.compute_tau_sc(DM, f)
    w_eff = np.sqrt(w_int**2 + tau_sc**2 + tau_DM**2 + t_samp**2)

    return w_eff


def flux_radio_obs(
    S_radio_f: np.ndarray,
    w_int: np.ndarray,
    w_eff: np.ndarray,
) -> np.ndarray:
    """
    Compute the flux density received by the telescope after taking into account that the pulse has been
    broadened by the propagation in the interstellar medium. We assume that the total fluence = S_radio_f x w_int
    is conserved as the pulse propagates in the interstellar medium. Since the pulse is broadened as it propagates,
    the flux received on Earth is given by S_radio_obs = fluence / w_eff, therefore S_radio_obs < S_radio_f.
    Also, since in our simulation we are assuming a simple squared pulse shape, the flux density computed here is equal
    to the peak flux density.

    Args:
        S_radio_f (np.ndarray): Intrinsic pulsar radio flux in [Jy].
        w_int (np.ndarray): Intrinsic pulse width in [rad].
        w_eff (np.ndarray): Effective pulse width in [rad].

    Returns:
        (np.ndarray): Observed pulsar radio flux in [Jy].
    """

    # Compute the total fluence.
    fluence_f = S_radio_f * w_int

    # Compute the observed radio flux in [Jy].
    S_radio_f_obs = fluence_f / w_eff

    return S_radio_f_obs


def flux_radio_obs_period_average(
    S_radio_f_obs: np.ndarray,
    P: np.ndarray,
    w_eff: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean flux density received by the telescope averaged over a spin period.
    We are assuming a simple squared pulse shape.

    Args:
        S_radio_f_obs (np.ndarray): Observed pulsar radio flux in [Jy].
        P (np.ndarray): Spin period of the pulsar in [s].
        w_eff (np.ndarray): Effective pulse width in [s].

    Returns:
        (np.ndarray): Observed pulsar radio flux averaged over a period in [Jy].
    """

    # Compute the observed radio flux in [Jy].
    S_radio_f_obs_mean = S_radio_f_obs * w_eff / P

    return S_radio_f_obs_mean


def sky_temperature_approx(
    l_gal: np.ndarray, b_gal: np.ndarray, f: float
) -> np.ndarray:
    """
    Sky temperature as a function of Galactic longitude and latitude (l, b) and frequency f.
    This function uses an empirical fit from Narayan (1987) and rescale to the given frequency using
    a relation from Johnston et al. (1992) (see also eq. (5) in Yusifov & Küçük 2004).

    Args:
        l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
        f (np.ndarray): Central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): Measured sky temperature in [K] as a function of the Galactic coordinates at frequency f.
    """

    # Sky temperature at 408 MHz from Narayan (1987).
    T_sky_400 = 25.0 + 275.0 / (
        (1.0 + (l_gal / 42.0) ** 2) * (1.0 + (b_gal / 3.0) ** 2)
    )

    # Rescale to the wanted frequency assuming a sky temperature spectral index of -2.6
    # (see Lawson et al. 1987, Johnston et al. 1992).
    T_sky_f = T_sky_400 * (408.0e6 / f) ** 2.6

    return T_sky_f


def sky_temperature_H81(
    l_gal: np.ndarray, b_gal: np.ndarray, f: float
) -> np.ndarray:
    """
    Sky temperature as a function of Galactic longitude and latitude (l, b) and frequency f.
    This function implements the sky temperature map at 408 MHz from Haslam et al. (1981), downloadable here:
    https://lambda.gsfc.nasa.gov/product/foreground/haslam_408.cfm.

    Args:
        l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
        f (np.ndarray): Central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): Measured sky temperature in [K] as a function of the Galactic coordinates at frequency f.
    """

    # Read the sky temperature map.
    file = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/Tsky_map_haslam81.fits",
    )
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

    # Rescale to the wanted frequency assuming a sky temperature spectral index of -2.6
    # (see Lawson et al. 1987, Johnston et al. 1992).
    T_sky_f = T_sky_400 * (408.0e6 / f) ** 2.6

    return T_sky_f


def sky_temperature_H81refined(
    l_gal: np.ndarray, b_gal: np.ndarray, f: float
) -> np.ndarray:
    """
    Sky temperature as a function of Galactic longitude and latitude (l, b) and frequency f.
    This function implements the sky temperature map at 408 MHz from Remazeilles et al. (2015) which is a
    refinement of the map from Haslam et al. (1981).
    The map is downloadable here:
    https://lambda.gsfc.nasa.gov/product/foreground/fg_2014_haslam_408_get.html.

    Args:
        l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
        f (np.ndarray): Central frequency at which the observation is performed [Hz].

    Returns:
        (np.ndarray): Measured sky temperature in [K] as a function of the Galactic coordinates at frequency f.
    """

    # Read the sky temperature map.
    file = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/multiband_surveys/Tsky_map_haslam81_refined.fits",
    )

    T_sky_map = hp.read_map(file, dtype=np.float64)

    # Convert coordinates into astropy coordinates object.
    coord = SkyCoord(l_gal, b_gal, frame="galactic", unit="deg")

    # Convert sky coordinates into pixel coordinates and extract the temperatures.
    l_g = coord.l.degree
    b_g = coord.b.degree
    vec = hp.rotator.dir2vec(l_g, b_g, lonlat=True)
    n_side = 512
    pix = hp.pixelfunc.vec2pix(n_side, vec[0], vec[1], vec[2], nest=False)

    T_sky_400 = T_sky_map[pix]

    # Rescale to the wanted frequency assuming a sky temperature spectral index of -2.6
    # (see Lawson et al. 1987, Johnston et al. 1992).
    T_sky_f = np.atleast_1d(T_sky_400 * (408.0e6 / f) ** 2.6)

    return T_sky_f


class SurveyRadio:
    """
    Class for any radio survey with a Gaussian telescope beam pattern.
    The parameters for the survey are imported from a JSON file.
    """

    def __import_parameters(self, parameters_path: str) -> None:
        """
        This routine imports the parameters of a radio survey.

        Args:
            parameters_path (str): Path to the survey_parameter.json file
                containing the parameters of the radio survey.
        """

        # Load parameters from JSON file.
        with open(parameters_path) as read_file:
            self.parameters = json.load(read_file)

        # Save the parameters.
        # deg_factor (float): degradation factor.
        # G0 (float): gain at the beam center [KJy ^ (-1)].
        # t_obs (float): integration time [s].
        # t_samp (float): sampling time [s].
        # T_sys (float): system temperature [K].
        # nu_central (float): central frequency of the bandwidth [Hz].
        # BW (float): frequency bandwidth [Hz].
        # channel_width (float): width of a single frequency channel [Hz].
        # n_pol (float): number of polarizations.
        # FWHM (float): FWHM of the beam[arcmin].
        # SNR_th (float): threshold signal-to-noise ratio.
        # RA_range (np.ndarray): range of the sky covered by the survey in RA [deg].
        # DEC_range(np.ndarray): range of the sky covered by the survey in DEC [deg].
        # l_range(np.ndarray): range of the sky covered by the survey in Galactic longitude l[deg].
        # b_range_abs(np.ndarray): absolute value of the range of the sky covered
        #   by the survey in Galactic latitude b [deg].
        self.deg_factor = self.parameters["deg_factor"]
        self.G0 = self.parameters["G0"]
        self.t_obs = self.parameters["t_obs"]
        self.t_samp = self.parameters["t_samp"]
        self.T_sys = self.parameters["T_sys"]
        self.f_central = self.parameters["f_central"]
        self.BW = self.parameters["BW"]
        self.channel_width = self.parameters["channel_width"]
        self.n_pol = self.parameters["n_pol"]
        self.FWHM = self.parameters["FWHM"]
        self.SNR_th = self.parameters["SNR_th"]
        self.RA_range = self.parameters["RA_range"]
        self.DEC_range = self.parameters["DEC_range"]
        self.l_range = self.parameters["l_range"]
        self.b_range_abs = self.parameters["b_range_abs"]
        self.name = self.parameters["name"]

    def __init__(
        self,
        parameters_path: str,
    ) -> None:
        """
        Radio survey initialization.

        Args:
            parameters_path (str): Path to the survey_parameter.json file
                containing the parameters of the radio survey.
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
            RA (np.ndarray): Right ascension in [deg] defined between [0, 360] deg in ICRS frame.
            DEC (np.ndarray): Declination in [deg] defined between [-90, 90] deg in ICRS frame.
            l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.

        Returns:
            (np.ndarray): Array of boolean variables: true if the pulsar is in the covered sky region, false if not.
        """
        if self.name == "HTRU high":

            coverage = (
                (RA > self.RA_range[0])
                & (RA < self.RA_range[1])
                & (DEC > self.DEC_range[0])
                & (DEC < self.DEC_range[1])
                & (
                    (
                        (
                            (l_gal > self.l_range[0][0])
                            & (l_gal < self.l_range[0][1])
                        )
                        | (
                            (l_gal > self.l_range[1][0])
                            & (l_gal < self.l_range[1][1])
                        )
                    )
                    | (
                        (np.abs(b_gal) > self.b_range_abs[0])
                        & (np.abs(b_gal) < self.b_range_abs[1])
                    )
                )
            )

        else:

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
        A Gaussian beam pattern is assumed to account for sensitivity decay for off-center detections.
        See Lorimer et al. (1993) and the paragraph following eq. (29) in Bates et al. (2014).

        Args:
            n_detection (int): Number of detections to simulate.

        Returns:
            (np.ndarray): Square of the offset from the beam center for each detection in [arcmin^2] .
        """
        offset2 = np.random.uniform(0.0, self.FWHM**2 / 4.0, n_detection)
        return offset2

    def gain_gaussian_beam(self, offset2: np.ndarray) -> np.ndarray:
        """
        This method simulates the gain pattern of a receiver.
        A Gaussian beam pattern is assumed to account for sensitivity decay for off-center detections.
        See Lorimer et al. (1993) and the paragraph following eq. (29) in Bates et al. (2014).

        Args:
            offset2 (np.ndarray): Squared offset from the beam center in [arcmin^2].

        Returns:
            (np.ndarray): Gain of the telescope for the given offset in [K Jy^(-1)].
        """
        G = self.G0 * np.exp(-2.77 * offset2 / self.FWHM**2)

        return G

    def radiometer_equation(
        self,
        S_radio_obs_mean: np.ndarray,
        G: np.ndarray,
        w_eff: np.ndarray,
        P: np.ndarray,
        T_sky: np.ndarray,
    ) -> np.ndarray:
        """
        Radiometer equation used to compute the signal-to-noise ratio of each pulsar given the observed
        period-averaged radio flux at a given frequency, the effective pulse width, the spin period and
        the survey parameters (see eq. (A1.22) in Lorimer & Kramer 2005). We are assuming a square pulse
        shape for simplicity with height equal to the observed flux and width equal to the effective width.

        Args:
            S_radio_obs_mean (np.ndarray): Observed period-averaged radio flux density in [Jy].
            G (np.ndarray): Gain of the telescope for the given detection in [K Jy^(-1)].
            w_eff (np.ndarray): Effective pulse width in [s].
            P (np.ndarray): Spin period in [s].
            T_sky (np.ndarray): Sky temperature for every detection in [K].

        Returns:
            (np.ndarray): Signal-to-noise ratio of the detection.
        """
        SNR = np.zeros(len(S_radio_obs_mean))

        # If the effective pulse width is larger than the spin period,
        # emission is continuous and the neutron star cannot be detected as a pulsar.
        cond = w_eff < P

        # Compute the SNR of each detection using the radiometer equation.
        SNR[cond] = (
            S_radio_obs_mean[cond]
            * G[cond]
            * np.sqrt(self.n_pol * self.t_obs * self.BW)
            * np.sqrt((P[cond] - w_eff[cond]) / w_eff[cond])
            / (self.deg_factor * (self.T_sys + T_sky[cond]))
        )

        return SNR

    def simulate_detection(
        self,
        S_radio_obs_mean: np.ndarray,
        l_gal: np.ndarray,
        b_gal: np.ndarray,
        w_eff: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate a detection: if the measured SNR surpasses the threshold SNR_th of the survey
        then the pulsar is detected.

        Args:
            S_radio_obs_mean (np.ndarray): Observed period-averaged radio flux density in [Jy].
            l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
            w_eff (np.ndarray): Effective pulse width in [s].
            P (np.ndarray): Spin period in [s].

        Returns:
            (np.ndarray): Array of boolean variables: true if the pulsar is detected,
                false if not.
        """

        # Store the total number of sources.
        n = len(S_radio_obs_mean)

        # Draw a random offset from the telescope beam center.
        offset2 = self.detection_offset(n)
        # Compute the gain corresponding to the offset detections.
        G = self.gain_gaussian_beam(offset2)

        # Compute the sky temperature in the coordinates of each detection at the central frequency of the survey.
        # We choose here to use the refined map from Remazeilles et al. (2015).
        T_sky = sky_temperature_H81refined(l_gal, b_gal, self.f_central)
        SNR_detection = self.radiometer_equation(
            S_radio_obs_mean, G, w_eff, P, T_sky
        )

        detected = SNR_detection > self.SNR_th

        return detected

    def detected_radio_population(
        self,
        w_int_s: np.ndarray,
        DM: np.ndarray,
        P: np.ndarray,
        age: np.ndarray,
        coverage: np.ndarray,
        l_gal: np.ndarray,
        b_gal: np.ndarray,
        S_radio_bol,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Compute the pulsars detected by a survey.
        This function is used in the simulate_population_magrot_det.py script.

        Args:
            w_int_s (np.ndarray): Intrinsic pulse widths in [s]
            DM (np.ndarray): Dispersion measure in [pc cm^-3].
            P (np.ndarray): Array of spin periods of the pulsars in [s].
            age (np.ndarray): Array of neutron star ages [yrs].
            coverage (np.ndarray): Array of indexes of the pulsar within the sky coverage.
            l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
            S_radio_bol (np.ndarray): Pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple consisting of three arrays defining the indexes of the
                pulsars detected by the survey, the effective pulse width and period-averaged flux.
        """

        # Computing the intrinsic radio flux density in [Jy].
        S_radio_f = er.flux_density_radio(
            S_radio_bol,
            f=self.f_central,
        )

        # Compute the effective pulse width in [s].
        w_eff = effective_pulse_width(
            w_int_s,
            DM,
            self.channel_width,
            self.f_central,
            self.t_samp,
        )

        # Compute the observed radio flux in [Jy].
        S_radio_obs = flux_radio_obs(S_radio_f, w_int_s, w_eff)

        # Compute the period-averaged flux in [Jy].
        S_radio_obs_mean = flux_radio_obs_period_average(S_radio_obs, P, w_eff)

        detected_radio = np.zeros(len(age), dtype=bool)

        detected_radio[coverage] = self.simulate_detection(
            S_radio_obs_mean[coverage],
            l_gal[coverage],
            b_gal[coverage],
            w_eff[coverage],
            P[coverage],
        )

        return detected_radio, w_eff, S_radio_obs_mean

    def detected_radio_population_full(
        self,
        w_int_s: np.ndarray,
        DM: np.ndarray,
        P: np.ndarray,
        l_gal: np.ndarray,
        b_gal: np.ndarray,
        S_radio_bol,
        intercepted_radio: np.ndarray,
        coverage_survey: np.ndarray,
        dist_cutoff: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
        Compute the pulsars detected by each survey.
        This function is used in the simulate_population_full.py script.

        Args:
            w_int_s (np.ndarray): Intrinsic pulse widths in [s]
            DM (np.ndarray): Dispersion measure in [pc cm^-3].
            P (np.ndarray): Array of spin periods of the pulsars in [s].
            l_gal (np.ndarray): Galactic longitude in [deg] defined between [-180, 180] deg.
            b_gal (np.ndarray): Galactic latitude in [deg] defined between [-90, 90] deg.
            S_radio_bol (np.ndarray): Pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].
            intercepted_radio: (np.ndarray) Array of boolean variables where true values represent stars that its
                beam crosses our line of sight.
            coverage_survey: (np.ndarray) Array of boolean variables where true values represent stars within the
                sky coverage of each survey.
            dist_cutoff: (np.ndarray) Array of boolean variables where true values represent stars within 35 kpc.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Tuple consisting of four arrays defining
                the indexes of the pulsars detected by the survey, the observed period-averaged radio flux density,the
                effective pulse width and the observed radio flux density.
        """

        detectable_radio_survey = (
            intercepted_radio & coverage_survey & dist_cutoff
        )

        # Computing the intrinsic radio flux density in [Jy].
        S_radio_f = np.zeros(cfg["NS_number"])
        S_radio_f[detectable_radio_survey] = er.flux_density_radio(
            S_radio_bol[detectable_radio_survey],
            f=self.f_central,
        )

        # Compute the effective pulse width in [s]
        w_eff = np.zeros(cfg["NS_number"])
        w_eff[detectable_radio_survey] = effective_pulse_width(
            w_int_s[detectable_radio_survey],
            DM[detectable_radio_survey],
            self.channel_width,
            self.f_central,
            self.t_samp,
        )

        # Compute the observed radio flux in [Jy].
        S_radio_obs = np.zeros(cfg["NS_number"])
        S_radio_obs[detectable_radio_survey] = flux_radio_obs(
            S_radio_f[detectable_radio_survey],
            w_int_s[detectable_radio_survey],
            w_eff[detectable_radio_survey],
        )

        # Compute the period-averaged flux in [Jy].
        S_radio_obs_mean = flux_radio_obs_period_average(S_radio_obs, P, w_eff)

        detected_radio_survey = np.zeros(cfg["NS_number"], dtype=bool)

        detected_radio_survey[
            detectable_radio_survey
        ] = self.simulate_detection(
            S_radio_obs_mean[detectable_radio_survey],
            l_gal[detectable_radio_survey],
            b_gal[detectable_radio_survey],
            w_eff[detectable_radio_survey],
            P[detectable_radio_survey],
        )

        return (
            detected_radio_survey,
            S_radio_obs_mean,
            w_eff,
            S_radio_obs,
        )
