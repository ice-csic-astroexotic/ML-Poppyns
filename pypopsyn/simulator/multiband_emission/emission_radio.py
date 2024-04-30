"""
Models for the pulsars' radio beam geometry and luminosity.

Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)

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

import pypopsyn.simulator.basics.constants as const
import pypopsyn.simulator.basics.random_sampler as rs
import pypopsyn.simulator.interstellar_medium.e_density_model as edm
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
from pypopsyn.simulator.config_simulator import cfg


def beam_aperture(P: np.ndarray, r_em: float) -> np.ndarray:
    """
    Half angular aperture in [rad] of the radio beam as a function of the spin period.
    It can be derived by assuming that the radio beam width extends in the open
    field line region around the magnetic poles of the star.
    See eq. (3.29) in Lorimer and Kramer (2005) and eq. (2) in Johnston et al. (2020)
    for a derivation and discussion of this model.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        r_em (float): distance from the center of the star where the radio emission
        is supposed to be generated [cm].

    Returns:
        (np.ndarray): half angular aperture of the radio beam in [rad].
    """

    rho_b = np.sqrt(9.0 * np.pi * r_em / (2.0 * const.C * P))

    return rho_b


def pulse_width(
    chi: np.ndarray, rho_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Formula to evaluate the intrinsic pulse width in [rad] from the radio beam angular aperture.
    This assumes that the line of sight intercepts the radio beam with an angle beta with
    respect to the center of the beam, so that -rho_b < beta < rho_b, with rho_b the angular
    aperture of the beam. See eq. (1) in Maciesiak et al. (2011a) and eq. (3.26) in Lorimer & Kramer (2004).

    Args:
        chi (np.ndarray): array of inclination angles between the magnetic axis and the rotation axis [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angles of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of intrinsic pulse widths in [rad].
    """

    # Compute the impact parameter, i.e., the angular distance between the LOS intercept and the center of the beam.
    beta = los - chi

    # Compute the quantity sin(w/4) where w is the pulse width.
    sin_w_4 = np.array(
        np.sqrt(
            (np.sin(rho_b / 2.0) ** 2 - np.sin(beta / 2) ** 2)
            / (np.sin(chi + beta) * np.sin(chi))
        )
    )
    # If sin(w/4) > 1 set w = 2*np.pi.
    sin_w_4[sin_w_4 > 1] = 1.0

    w_int = 4.0 * np.arcsin(sin_w_4)

    return w_int


def solid_angle_radio_beams(rho_b: np.ndarray) -> np.ndarray:
    """
    Total solid angle covered by the two radio beams as a function of the radio beam aperture.

    Args:
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].

    Returns:
        (np.ndarray): total solid angle covered by the radio beams.
    """

    solid_angle = 4 * np.pi * (1 - np.cos(rho_b))

    return solid_angle


def beam_fraction(chi: np.ndarray, rho_b: np.ndarray) -> np.ndarray:
    """
    Fraction of solid angle covered by the radio beam in an entire pulsar rotation
    as a function of the inclination angle chi and the radio beam aperture.
    This is also the probability that the radio beam intercepts our line of sight.
    See Appendix in Emmering and Chevalier (1989).

    Args:
        chi (np.ndarray): array of inclination angles of the magnetic axis
        with respect to the rotation axis of the pulsars in [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].

    Returns:
        (np.ndarray): fraction of solid angle swept by the radio beam.
    """

    beam_frac = np.cos(np.maximum(0, (chi - rho_b))) - np.cos(
        np.minimum(np.pi / 2.0, (chi + rho_b))
    )

    return beam_frac


def los_intercept(
    chi: np.ndarray, rho_b: np.ndarray, los: np.ndarray
) -> np.ndarray:
    """
    Evaluate if the radio beam intercepts the line of sight (LOS), assuming random
    orientation of the LOS with respect to the rotation axis.

    Args:
        chi (np.ndarray): array of inclination angles of the pulsars in [rad].
        rho_b (np.ndarray): array of angular apertures of the radio beam of the pulsars in [rad].
        los (np.ndarray): polar angle of the line of sight intercept computed with respect
        to the rotation axis of the star [rad].

    Returns:
        (np.ndarray): array of boolean variables: true if the radio beam intercepts the LOS and false if not.
    """

    condition = (los > chi - rho_b) & (los < chi + rho_b)
    intercepted = np.array(condition)

    return intercepted


def pdf_luminosity_radio(P: np.ndarray, P_dot: np.ndarray) -> np.ndarray:
    """
    Draw random bolometric radio luminosities from a distribution that depends on the spin period
    and spin period derivative. We assume a random log-normal spread for the normalization constant L_0.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].

    Returns:
        (np.ndarray): pulsar radio luminosity [erg s^(-1)] drawn from a log-normal distribution.
    """

    NS_number = len(P)

    L_0 = 10 ** np.random.normal(
        cfg["L_radio_log10_mean"], cfg["L_radio_log10_sigma"], NS_number
    )
    L_radio = L_0 * (P ** (-3) * P_dot) ** cfg["epsilon_L"]

    return L_radio


def flux_radio(
    L_radio: np.ndarray,
    d: np.ndarray,
    solid_angle: np.ndarray,
) -> np.ndarray:
    """
    Compute the intrinsic bolometric radio flux for each pulsars in [erg s^(-1) cm^(-2)].

    Args:
        L_radio (np.ndarray): pulsar bolometric radio luminosity in [erg s^(-1)].
        d (np.ndarray): distance to the pulsar in [kpc].
        solid_angle (np.ndarray): solid angle covered by the radio beams in [sr].

    Returns:
        (np.ndarray): intrinsic pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].
    """

    # Convert distance from [kpc] to [cm].
    d_cm = d * const.KPC_TO_CM

    S_radio = L_radio / (solid_angle * d_cm**2)

    return S_radio


def flux_density_radio(
    S_radio_bol: np.ndarray,
    f: float,
    spectral_index: float = -1.6,
    f_min: float = 1.0e7,
    f_max: float = 1.0e11,
) -> np.ndarray:
    """
    Compute the radio flux density at a given frequency f assuming a power law spectral shape for the radio emission.

    Args:
        S_radio_bol (np.ndarray): pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].
        f (float): frequency in [Hz] at which the radio luminosity has to be computed.
        spectral_index (float): spectral index of the radio emission, assuming a power-law spectrum.
        f_min (float): frequency lower limit of the radio emission spectrum [Hz].
        f_max (float): frequency upper limit of the radio emission spectrum [Hz].

    Returns:
        (np.ndarray): intrinsic pulsar radio flux density in [Jy] at the frequency f.
    """

    S_radio_f = (
        (spectral_index + 1)
        * S_radio_bol
        / (f_max ** (spectral_index + 1) - f_min ** (spectral_index + 1))
        * f**spectral_index
    ) / const.JY_TO_ERG

    return S_radio_f


def calculate_radio_emission(
    P: np.ndarray,
    age: np.ndarray,
    l_gal: np.ndarray,
    b_gal: np.ndarray,
    dist: np.ndarray,
    B: np.ndarray,
    chi: np.ndarray,
    idx_det,
) -> dict:
    """
    Compute the radio beam geometry, the intrinsic bolometric radio flux and the DM.
    Note that the luminosity and DM are computed only for those pulsars whose beams cross our line of sight.
    This function is used in the simulate_population_magrot_det.py script.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        age (np.ndarray): array of neutron star ages [yrs].
        l_gal (np.ndarray): array of galactic longitudes in [deg] defined between [-180, 180] deg.
        b_gal (np.ndarray): array of galactic latitudes in [deg] defined between [-90, 90] deg.
        dist (np.ndarray): array of distances from the ICRS origin in [kpc].
        B (np.ndarray): array of neutron stars' final magnetic field strengths in [G].
        chi (np.ndarray): array of the misalignment angles in [rad].
        idx_det (np.ndarray): array of the indexes of detected pulsars.

    Returns:
        (Dict): dictionary with the intrinsic properties of the pulsars whose beam crosses our line of sight.
    """

    # Determining the radio beam angular aperture.
    rho_beam = beam_aperture(P, cfg["r_em"])

    # Determining the solid angle covered by the two radio beams.
    solid_angle_beam = solid_angle_radio_beams(rho_beam)

    # Drawing a random angular intercept for the line of sight.
    # Note that since we assume symmetry between the northern and southern hemisphere of the star
    # we only need to consider one hemisphere, e.g., the northern one.
    los_grid = np.linspace(0.0, np.pi / 2, cfg["resolution"])
    los_rand = rs.random_from_pdf(los_grid, np.sin, len(age))

    # Determining if the pulsar's radio beam intercepts our line of sight.
    intercepted_radio = los_intercept(
        chi,
        rho_beam,
        los_rand,
    )

    # Select only neutron stars that point at us.
    idx_det = idx_det[intercepted_radio]

    age_det = age[intercepted_radio]
    l_det = l_gal[intercepted_radio]
    b_det = b_gal[intercepted_radio]
    dist_det = dist[intercepted_radio]
    B_det = B[intercepted_radio]
    chi_det = chi[intercepted_radio]
    P_det = P[intercepted_radio]
    rho_beam_det = rho_beam[intercepted_radio]
    los_rand_det = los_rand[intercepted_radio]
    solid_angle_beam = solid_angle_beam[intercepted_radio]

    # Determining the final period derivative.
    period_derivative_vect = np.vectorize(pdv.period_derivative)
    P_dot_det = (
        period_derivative_vect(
            B_det,
            chi_det,
            P_det,
        )
        / const.YR_TO_S
    )

    # Determining the bolometric radio luminosity.
    L_radio_bol = pdf_luminosity_radio(P_det, P_dot_det)

    # Computing the intrinsic bolometric radio flux.
    S_radio_bol = flux_radio(
        L_radio_bol,
        dist_det,
        solid_angle_beam,
    )

    # Computing the intrinsic pulse width of the radio pulse.
    w_int = pulse_width(
        chi_det,
        rho_beam_det,
        los_rand_det,
    )

    # Convert pulse width from [rad] to [s].
    w_int_s = w_int * P_det / (2.0 * np.pi)

    # Computing the DM.
    DM = edm.compute_DM(
        l_det,
        b_det,
        dist_det,
        cfg["ed_model"],
    )

    dictionary_intercepted_radio = {
        "age_det": age_det,
        "l_det": l_det,
        "b_det": b_det,
        "B_det": B_det,
        "chi_det": chi_det,
        "P_det": P_det,
        "P_dot_det": P_dot_det,
        "w_int_s": w_int_s,
        "L_radio_bol": L_radio_bol,
        "S_radio_bol": S_radio_bol,
        "DM": DM,
        "idx_det": idx_det,
        "intercepted_radio": intercepted_radio,
    }

    return dictionary_intercepted_radio


def calculate_radio_emission_full(
    P: np.ndarray,
    P_dot: np.ndarray,
    dist: np.ndarray,
    chi: np.ndarray,
):
    """
    Compute the radio beam geometry and the intrinsic bolometric radio flux. This function is only used in the
    simulate_population_full.py script, where we perform the dynamical and magneto-rotational evolution together.

    Args:
        P (np.ndarray): array of spin periods of the pulsars in [s].
        P_dot (np.ndarray): array of spin period derivatives of the pulsars in [s/s].
        dist (np.ndarray): array of distances from the ICRS origin in [kpc].
        chi (np.ndarray): array of the misalignment angles in [rad].

    Returns:
        intercepted_radio (np.ndarray): array of Booleans with the pulsars whose beams cross our line of sight.
        S_radio_bol (np.ndarray): pulsar bolometric radio flux in [erg s^(-1) cm^(-2)].
        w_int_s (np.ndarray): intrinsic pulse widths in [s].
        L_radio_bol (np.ndarray): pulsar radio luminosity [erg s^(-1)] drawn from a log-normal distribution.
    """

    # Determining the luminosity in different electromagnetic bands.
    L_radio_bol = pdf_luminosity_radio(
        P,
        P_dot,
    )

    # Determining the radio beam angular aperture.
    rho_beam = beam_aperture(P, cfg["r_em"])

    # Determining the solid angle covered by the two radio beams.
    solid_angle_beam = solid_angle_radio_beams(rho_beam)

    # Drawing a random angular intercept for the line of sight.
    # Note that since we assume symmetry between the northern and southern hemisphere of the star
    # we only need to consider one hemisphere, e.g., the northern one.
    los_grid = np.linspace(0.0, np.pi / 2, cfg["resolution"])
    los_rand = rs.random_from_pdf(los_grid, np.sin, cfg["NS_number"])

    # Selecting the pulsars whose radio beam intercepts our line of sight.
    intercepted_radio = los_intercept(
        chi,
        rho_beam,
        los_rand,
    )

    # Computing the intrinsic pulse width of the radio pulse.
    w_int = np.zeros(cfg["NS_number"])
    w_int[intercepted_radio] = pulse_width(
        chi[intercepted_radio],
        rho_beam[intercepted_radio],
        los_rand[intercepted_radio],
    )
    # Convert pulse width from [rad] to [s].
    w_int_s = w_int * P / (2.0 * np.pi)

    # Computing the intrinsic bolometric radio flux.
    S_radio_bol = np.zeros(cfg["NS_number"])
    S_radio_bol[intercepted_radio] = flux_radio(
        L_radio_bol[intercepted_radio],
        dist[intercepted_radio],
        solid_angle_beam[intercepted_radio],
    )

    return intercepted_radio, S_radio_bol, w_int_s, L_radio_bol
