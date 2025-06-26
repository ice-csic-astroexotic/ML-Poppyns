"""
    Model for the pulsar X-ray surveys.

    We consider here the detection of thermally emitting neutron stars.


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
    This is used if we are only interested to filter neutron stars above a given X-ray flux threshold.

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold (float): The flux threshold for X-ray detection in [erg s^-1 cm^-2].

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
    Compute the pulsars detected by an X-ray survey with a given flux threshold from a gaussian distribution in log10
    to mimic all the uncertainties inherent in a detection with an instrument.

    Args:
        S_x (np.ndarray): Array of observed X-ray fluxes in [erg s^-1 cm^-2].
        S_x_threshold_log10_mean (float): The mean of the flux threshold distribution for X-ray detection in [erg s^-1 cm^-2].
        S_x_threshold_log10_sigma (float): The standard deviation of the flux threshold distribution for X-ray detection in
            [erg s^-1 cm^-2].

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
    Class for any Xray survey.
    The parameters for the survey are imported from a JSON file.
    """

    def __import_parameters(self, parameters_path: str) -> None:
        """
        This routine imports the parameters of a X-ray survey.

        Args:
            parameters_path (str): Path to the survey_parameter.json file
                containing the parameters of the X-ray survey.
        """

        # Load parameters from JSON file.
        with open(parameters_path) as read_file:
            self.parameters = json.load(read_file)

        # apply_sharp_flux_threshold (bool): boolean flag to apply a simplified survey with a sharp flux threshold or
        # not.
        self.apply_sharp_flux_threshold = self.parameters[
            "apply_sharp_flux_threshold"
        ]
        # sharp_flux_threshold (float): sharp flux threshold value for a generic survey.
        self.sharp_flux_threshold = self.parameters["sharp_flux_threshold"]
        # S_x_threshold_log10_mean_long_exposure (float): mean for the log10 of the
        # flux threshold distribution for a survey with short exposure.
        self.S_x_threshold_log10_mean_long_exposure = self.parameters[
            "S_x_threshold_log10_mean_long_exposure"
        ]
        # S_x_threshold_log10_sigma_long_exposure (float): standard deviation for the log10 of the
        # flux threshold distribution for a survey with long exposure.
        self.S_x_threshold_log10_sigma_long_exposure = self.parameters[
            "S_x_threshold_log10_sigma_long_exposure"
        ]
        # S_x_threshold_log10_mean_short_exposure (float): mean for the log10 of the
        # flux threshold distribution for a survey with short exposure.
        self.S_x_threshold_log10_mean_short_exposure = self.parameters[
            "S_x_threshold_log10_mean_short_exposure"
        ]
        # S_x_threshold_log10_sigma_short_exposure (float): standard deviation for the log10 of the
        # flux threshold distribution for a survey with short exposure.
        self.S_x_threshold_log10_sigma_short_exposure = self.parameters[
            "S_x_threshold_log10_sigma_short_exposure"
        ]
        # RA_range (np.ndarray): range of the sky covered by the survey in RA [deg].
        self.RA_range = self.parameters["RA_range"]
        # DEC_range(np.ndarray): range of the sky covered by the survey in DEC [deg].
        self.DEC_range = self.parameters["DEC_range"]
        # l_range(np.ndarray): range of the sky covered by the survey in Galactic longitude l[deg].
        self.l_range = self.parameters["l_range"]
        # b_range_abs(np.ndarray): absolute value of the range of the sky covered
        #   by the survey in Galactic latitude b [deg].
        self.b_range_abs = self.parameters["b_range_abs"]
        # name(str): name of the survey.
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

    def detected_xray_population(
        self,
        S_x: np.ndarray,
        outburst_mask: np.ndarray,
    ) -> np.ndarray:
        """
        This function detects neutron stars based on their X-ray fluxes and emulates X-ray surveys with different
        flux thresholds, and taking into account the detection bias of neutron stars that experience an outburst.

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
            # We assume that all neutron stars that go in outburst are detected.
            # This is because during outburst events fluxes increase and triggers X-ray and soft gamma-ray satellites like
            # Swift or Fermi. Usually Swift XRT is able to detect if there is a periodicity during the outburst event, when the
            # flux is still high and recognize the presence of a magnetar.
            # After the flux goes back to the quiescent state for the sources that went on outburst we emulate a deep
            # observations with relatively long exposure time to see if the quiescent emission is detectable.
            # For this we consider a flux threshold around 10^-14 erg s^-1 cm^-2 with an intrinsic dispersion to mimic the
            # detection sensitivity of instruments like XMM-Newton or Chandra with relatively long exposure times.
            detected_outburst_mask = smooth_flux_filter(
                S_x,
                self.S_x_threshold_log10_mean_long_exposure,
                self.S_x_threshold_log10_sigma_long_exposure,
            )

            # To include sources that didn't go in outburst but have high quiescent fluxes we also include a filter with
            # a higher flux threshold which is related to lower exposure time.
            # This emulates all-sky surveys like the one performed by ROSAT
            # or the slew mode in XMM that are more sensitive to brighter X-ray sources.
            detected_bright_mask = smooth_flux_filter(
                S_x,
                self.S_x_threshold_log10_mean_short_exposure,
                self.S_x_threshold_log10_sigma_short_exposure,
            )
            detected_xray_mask = (
                outburst_mask & detected_outburst_mask
            ) | detected_bright_mask

        return detected_xray_mask
