"""
    Model for the pulsar X-ray surveys.

    We consider here the detection of thermally emitting neutron stars, by considering two types of surveys:

    1) xray_flux_threshold: Detect neutron stars that have fluxes above a given threshold flux without considering
        any other detection biases.
    2) xray_realistic: Detect neutron stars taking into account their outburst probability and combining two flux
        filters to better reproduce the observed flux distribution of magnetars and XDINSs together.
        The first flux filter has a lower average flux threshold and applies to neutron stars that go into outburst.
        In the catalog of observed thermally emitting neutron stars, we only consider sources with detected quiescent
        thermal emission. I.e., we ignore those magnetars that were discovered in outburst but whose quiescent emission
        is too faint to be detected. This flux filter removes simulated neutron stars that despite going outburst have a
        quiescent emission that is too faint.
        The second flux filter has a higher average flux threshold and applies to neutron stars that are bright
        like XDINSs even if they did not show any outburst activity.

    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import json

import numpy as np


def sharp_flux_filter(
    S_x: np.ndarray,
    S_x_threshold: float = 1.0e-15,
) -> np.ndarray:
    """
    Compute the pulsars detected by an X-ray survey with a given sharp flux threshold.
    This is used if we are only interested in filtering neutron stars above a given X-ray flux threshold.

    As a default, the S_x_threshold parameter is set to be 1e-15 erg s^-1 cm^-2 (see Gullón et al. 2015).

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold (float): The flux threshold for X-ray detections in [erg s^-1 cm^-2].

    Returns:
        (np.ndarray): Boolean mask to select the pulsars detected above a given flux threshold.
    """

    # Filter the neutron stars according to a threshold flux.
    detected_x = S_x > S_x_threshold

    return detected_x


def smooth_flux_filter(
    S_x: np.ndarray,
    S_x_threshold_log10_mean: float = -15,
    S_x_threshold_log10_sigma: float = 0.5,
) -> np.ndarray:
    """
    Compute the pulsars detected by an X-ray survey with a given flux threshold drawn from a Gaussian distribution
    in log10 for each source, to mimic uncertainties inherent to detections with an X-ray telescope.

    As a default, the S_x_threshold_log10_mean parameter is set to be 1e-15 erg s^-1 cm^-2 (see Gullón et al. 2015).

    The value of S_x_threshold_log10_sigma is chosen to mimic the variation in the flux threshold due to different
    exposure times, background levels and instrument sensitivity (see for example Fig. 3 in Watson et al. 2001;
    The XMM-Newton Serendipitous Survey I. The role of XMM-Newton Survey Science Centre).

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold_log10_mean (float): The mean of the flux threshold distribution for X-ray detection
            in [erg s^-1 cm^-2].
        S_x_threshold_log10_sigma (float): The standard deviation of the flux threshold distribution for X-ray
            detections in [erg s^-1 cm^-2].

    Returns:
        (np.ndarray): Boolean mask to select the pulsars detected above a given flux threshold.
    """
    flux_threshold = 10 ** np.random.normal(
        S_x_threshold_log10_mean,
        S_x_threshold_log10_sigma,
        len(S_x),
    )
    detected_mask = S_x > flux_threshold
    return detected_mask


class SurveyXray:
    """
    Class for any X-ray survey.

    The parameters for the survey are imported from a JSON file.
    """

    def __import_parameters(self, parameters_path: str) -> None:
        """
        This routine imports the parameters of an X-ray survey.

        Args:
            parameters_path (str): Path to the survey_parameter.json file
                containing the parameters of the X-ray survey.
        """

        # Load parameters from JSON file.
        with open(parameters_path) as read_file:
            self.parameters = json.load(read_file)

        # apply_sharp_flux_threshold (bool): Boolean flag to apply a simplified survey with a sharp flux threshold or
        #   not.
        # sharp_flux_threshold (float): Sharp flux threshold value for a generic survey.
        # S_x_threshold_log10_mean_long_exposure (float): Mean for the log10 of the flux threshold distribution for a
        #   survey with short exposure.
        # S_x_threshold_log10_sigma_long_exposure (float): Standard deviation for the log10 of the flux threshold
        #   distribution for a survey with long exposure.
        # S_x_threshold_log10_mean_short_exposure (float): Mean for the log10 of the flux threshold distribution for
        #   a survey with short exposure.
        # S_x_threshold_log10_sigma_short_exposure (float): Standard deviation for the log10 of the flux threshold
        #   distribution for a survey with short exposure.
        # RA_range (np.ndarray): Range of the sky covered by the survey in RA [deg].
        # DEC_range(np.ndarray): Range of the sky covered by the survey in DEC [deg].
        # l_range(np.ndarray): Range of the sky covered by the survey in Galactic longitude l[deg].
        # b_range_abs(np.ndarray): Absolute value of the range of the sky covered by the survey in Galactic latitude
        #   b [deg].
        # name(str): Name of the survey.
        self.apply_sharp_flux_threshold = self.parameters[
            "apply_sharp_flux_threshold"
        ]
        self.sharp_flux_threshold = self.parameters["sharp_flux_threshold"]
        self.S_x_threshold_log10_mean_long_exposure = self.parameters[
            "S_x_threshold_log10_mean_long_exposure"
        ]
        self.S_x_threshold_log10_sigma_long_exposure = self.parameters[
            "S_x_threshold_log10_sigma_long_exposure"
        ]
        self.S_x_threshold_log10_mean_short_exposure = self.parameters[
            "S_x_threshold_log10_mean_short_exposure"
        ]
        self.S_x_threshold_log10_sigma_short_exposure = self.parameters[
            "S_x_threshold_log10_sigma_short_exposure"
        ]
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
        X-ray survey initialization.

        Args:
            parameters_path (str): Path to the survey_parameter.json file
                containing the parameters of the X-ray survey.
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
            (np.ndarray): Array of boolean variables: True if the pulsar is in the covered sky region, False if not.
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

    def detected_xray_population(
        self,
        S_x: np.ndarray,
        outburst_mask: np.ndarray,
    ) -> np.ndarray:
        """
        This function detects neutron stars based on their X-ray fluxes and emulates X-ray surveys with different
        flux thresholds, taking into account the detection biases of neutron stars that experience an outburst.

        Args:
            S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
            outburst_mask (np.ndarray): Boolean mask for outbursts events.

        Returns:
            (np.ndarray): Boolean mask to select the neutron stars detected by the survey.
        """

        if self.apply_sharp_flux_threshold:
            # Apply a sharp flux threshold to cut out very faint sources.
            detected_xray_mask = sharp_flux_filter(
                S_x,
                self.sharp_flux_threshold,
            )
        else:
            # If we do not apply a sharp flux threshold, we instead consider a detection bias towards neutron stars that go in outburst.
            # This is justified because many magnetars are discovered during outburst events.
            # While in outburst fluxes increase, triggering X-ray and soft gamma-ray satellites like Swift or Fermi.
            # Usually Swift XRT detects the presence of a magnetar, if the high-flux state during the outburst event
            # shows a periodicity.
            # Once the flux reduces back to quiescent levels following the outburst, we emulate follow-up of these sources
            # with more sensitive instruments such as XMM-Newton or Chandra using deep observations with relatively long
            # exposure time to see if the quiescent emission is detectable.
            # For this purpose, we consider a flux threshold of around 10^-14 [erg s^-1 cm^-2] with an intrinsic dispersion.
            # This choice has been made by eye to recover the low-flux part of the observed flux distribution.
            detected_outburst_mask = smooth_flux_filter(
                S_x,
                self.S_x_threshold_log10_mean_long_exposure,
                self.S_x_threshold_log10_sigma_long_exposure,
            )

            # To include sources that didn't go into outburst but have high quiescent fluxes, we also include a filter with
            # a higher flux threshold which is related to lower exposure time.
            # This emulates all-sky surveys like the one performed by ROSAT or the slew mode in XMM that are more
            # sensitive to brighter X-ray sources and can detect sources like the XDINSs.
            detected_bright_mask = smooth_flux_filter(
                S_x,
                self.S_x_threshold_log10_mean_short_exposure,
                self.S_x_threshold_log10_sigma_short_exposure,
            )
            detected_xray_mask = (
                outburst_mask & detected_outburst_mask
            ) | detected_bright_mask

        return detected_xray_mask
