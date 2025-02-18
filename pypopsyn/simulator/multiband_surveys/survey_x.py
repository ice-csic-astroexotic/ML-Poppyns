"""
    Model for the pulsar X-ray surveys.

    We consider here the detection of thermally emitting neutron stars.


    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
import pypopsyn.simulator.multiband_emission.emission_xray as ex


def detected_x_population(
    P: np.ndarray,
    B: np.ndarray,
    B_initial: np.ndarray,
    chi: np.ndarray,
    age: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    dist: np.ndarray,
    coverage: np.ndarray,
    L_x_interpolator: RectBivariateSpline,
    L_x_threshold: float,
    S_x_abs_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the pulsars detected by an X-ray survey with a given flux threshold.
    This function is used in the simulate_population_magrot_det.py script.

    Args:
        P (np.ndarray): Array of spin periods of the pulsars in [s].
        B (np.ndarray): Array of evolved magnetic fields of the pulsars in [G].
        B_initial (np.ndarray): Array of initial magnetic fields of the pulsars in [G].
        chi (np.ndarray): Array of inclination angles in [rad].
        age (np.ndarray): Array of neutron star ages [yrs].
        ra (np.ndarray): Right ascension in [deg] defined between [0, 360] deg in ICRS frame.
        dec (np.ndarray): Declination in [deg] defined between [-90, 90] deg in ICRS frame.
        dist (np.ndarray): Array of distances from the ICRS origin in [kpc].
        coverage (np.ndarray): Array of boolean variables indicating the pulsars within the sky coverage.
        L_x_interpolator (RectBivariateSpline): Interpolator used to calculate thermal X-ray luminosity based on age and magnetic field.
        L_x_threshold (float): A lower limit for the X-ray luminosity.
        S_x_abs_threshold (float): The absorbed flux threshold for X-ray detection.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Tuple containing the following arrays:
            - Boolean mask to select the pulsars detected by the survey.
            - X-ray thermal luminosities in [erg/s].
            - X-ray absorbed fluxes in [erg s^-1 cm^-2].
            - N_H column density in [cm^-2].
            - Spin period derivative values in [s/s].
    """

    S_x_abs = np.zeros(len(age))
    N_H = np.zeros(len(age))

    # Determining the final period derivative.
    period_derivative_vect = np.vectorize(pdv.period_derivative)
    P_dot = (
        period_derivative_vect(
            B,
            chi,
            P,
        )
        / const.YR_TO_S
    )

    L_x_therm = L_x_interpolator.ev(age, B_initial)

    # Select only the stars that have sufficiently high luminosity and fall in the X-ray survey coverage.
    L_x_mask = L_x_therm > L_x_threshold
    coverage = coverage & L_x_mask

    # Compute the absorbed fluxes and the N_H column density.
    S_x_abs[coverage], N_H[coverage] = ex.flux_xray_absorbed(
        L_x_therm[coverage],
        B[coverage],
        ra[coverage],
        dec[coverage],
        dist[coverage],
    )

    # Filter the neutron stars according to a threshold flux.
    detected_x = S_x_abs > S_x_abs_threshold

    return detected_x, L_x_therm, S_x_abs, N_H, P_dot
