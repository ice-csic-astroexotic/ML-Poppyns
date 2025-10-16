"""
    Loader for catalogs with neutron star data.

    This module contains methods to load the data of the observed radio pulsar population from
    the ATNF Pulsar Catalogue, the TPA program on MeerKat presented in Posselt et al. (2023) and
    the catalog of thermally emitting X-ray neutron stars.

    In the following, we consider the fluxes from the ch6flux column in Posselt et al. (2023),
    which correspond to measurements at 1429 MHz. These fluxes are used in Figure 7
    of the paper for comparison with those from the ATNF Pulsar Catalogue.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo  (pardo@ice.csic.es)
"""

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd


def load_atnf_meerkat_catalog(
    path_atnf_catalog: pathlib.Path,
    path_meerkat_catalog: pathlib.Path,
) -> Tuple[dict, dict]:
    """
    Load the ATNF Pulsar Catalogue and the radio flux data from the TPA program on MeerKat and return dictionaries with
    the properties of detected neutron stars for the following radio surveys:

        1) PMPS: the Parkes Multibeam Pulsar Survey (see Manchester et al. 2001, Lorimer et al. 2006)
        2) SMPS: the Swinburne Parkes Multibeam Pulsar Survey (see Edwards et al. 2001, Jacoby et al. 2009)
        3) HTRU: the High Time Resolution Universe Survey (see Keith et al. 2010)

    Args:
        path_atnf_catalog (pathlib.Path): Path to the ATNF Pulsar Catalogue.
        path_meerkat_catalog (pathlib.Path): Path to the TPA Program with MeerKat catalog.

    Returns:
        (Tuple[dict, dict]): A tuple object containing the following dictionaries:

            - A dictionary with the properties of detected neutron stars in the ATNF catalog for each radio survey.
            - A dictionary with the properties of detected neutron stars in the TPA program for each radio survey.
    """

    # Initialize the dictionaries where the filtered properties for each survey will be saved.
    surveys_atnf = {
        "PMPS": {},
        "SMPS": {},
        "HTRU_low-mid": {},
    }
    surveys_meerkat = {
        "PMPS": {},
        "SMPS": {},
        "HTRU_low-mid": {},
    }

    # Read the full ATNF catalogue.csv file. Binary pulsars are already excluded.
    df_atnf = pd.read_csv(
        path_atnf_catalog,
        delimiter=";",
        header=[0, 1],
    )

    df_atnf.columns = df_atnf.columns.get_level_values(0)

    # Read in the pulsar data from the TPA program.
    df_meerkat = pd.read_csv(
        path_meerkat_catalog,
        delimiter=",",
    )

    # Select only stars with measured P, Pdot, DM and radio flux.
    # We also select only those that are not in globular clusters or in the Magellanic Clouds.
    discard = [
        "EXGAL:SMC",
        "EXGAL:LMC",
        "GC:47Tuc",
        "GC:M3",
        "GC:M5",
        "GC:M13",
        "GC:NGC6440",
        "GC:Ter5",
        "GC:NGC6441",
        "GC:NGC6517",
        "GC:NGC6522",
        "GC:NGC6624",
        "GC:M28(NGC6626)",
        "GC:NGC6652",
        "GC:M22(NGC6656)",
        "GC:NGC6752",
        "GC:NGC6760",
        "GC:M15",
        "GC:M30",
    ]

    df_atnf = df_atnf[~df_atnf["ASSOC"].str.match("|".join(discard))]

    # Select only isolated, non-recycled neutron stars, i.e., those with P > 0.01 s and Pdot > 1e-19.
    df_atnf = df_atnf[df_atnf["P0"].to_numpy().astype(np.float64) > 0.01]
    df_atnf = df_atnf[
        (df_atnf["P1"].to_numpy().astype(np.float64) > 1.0e-19)
        | (df_atnf["P1"].isin(["NAN"]))
    ]

    # Parkes Multibeam Pulsar Survey database.
    df_atnf_pmps = df_atnf[df_atnf["SURVEY"].str.contains("pksmb")]

    # Select only pulsars falling into the PMPS sky coverage where completeness is above 90%.
    # See Lorimer et al. (2006) for details.

    # Extracting Galactic longitude, latitude.
    l_pmps_obs = df_atnf_pmps["Gl"].to_numpy().astype(np.float64)
    b_pmps_obs = df_atnf_pmps["Gb"].to_numpy().astype(np.float64)

    # Converting galactic longitude into the range [-180., 180].
    l_pmps_obs[(l_pmps_obs > 180.0) & (l_pmps_obs < 360.0)] = (
        l_pmps_obs[(l_pmps_obs > 180.0) & (l_pmps_obs < 360.0)] - 360.0
    )

    cond = (
        (l_pmps_obs > -100.0)
        & (l_pmps_obs < 50.0)
        & (np.abs(b_pmps_obs) < 5.0)
    )

    df_atnf_pmps = df_atnf_pmps[cond]

    # Save the properties in the PMPS dictionary.
    surveys_atnf["PMPS"]["RA"] = (
        df_atnf_pmps["RAJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["DEC"] = (
        df_atnf_pmps["DECJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["l_gal"] = l_pmps_obs[cond]
    surveys_atnf["PMPS"]["b_gal"] = (
        df_atnf_pmps["Gb"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["P"] = (
        df_atnf_pmps["P0"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["P_dot"] = (
        df_atnf_pmps["P1"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["DM"] = (
        df_atnf_pmps["DM"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["dist"] = (
        df_atnf_pmps["DIST"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["S1400"] = (
        df_atnf_pmps["S1400"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["pm_RA"] = (
        df_atnf_pmps["PMRA"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["pm_DEC"] = (
        df_atnf_pmps["PMDEC"].to_numpy().astype(np.float64)
    )
    surveys_atnf["PMPS"]["w10"] = (
        df_atnf_pmps["W10"].to_numpy().astype(np.float64)
    )

    # Merge the MeerKat TPA program data with the PMPS ATNF Pulsar Catalogue data to obtain MeerKAT flux measurements
    # for the PMPS pulsars.
    df_meerkat_pmps = pd.merge(
        df_meerkat, df_atnf_pmps, left_on="PSRJ", right_on="PSRJ"
    )
    df_meerkat_pmps = df_meerkat_pmps.dropna(subset=["ch6flux"])

    # Save the properties in the PMPS MeerKat dictionary.
    # We convert the MeerKat fluxes from [Jy] to [mJy] to compare with simulations.
    surveys_meerkat["PMPS"]["S1400"] = (
        df_meerkat_pmps["ch6flux"].to_numpy().astype(np.float64) / 1000
    )
    surveys_meerkat["PMPS"]["P"] = (
        df_meerkat_pmps["P0"].to_numpy().astype(np.float64)
    )
    surveys_meerkat["PMPS"]["P_dot"] = (
        df_meerkat_pmps["P1"].to_numpy().astype(np.float64)
    )

    # Swinburne Parkes Multibeam Pulsar Survey database.
    df_atnf_smps = df_atnf[df_atnf["SURVEY"].str.contains("pkssw")]

    # Select only pulsars falling into the SMPS sky coverage where completeness is above 90%.
    # See Edwards et al. (2001) and Jacoby et al. (2009) for details.

    # Extracting Galactic longitude.
    l_smps_obs = df_atnf_smps["Gl"].to_numpy().astype(np.float64)

    # Converting galactic longitude into the range [-180., 180].
    l_smps_obs[(l_smps_obs > 180.0) & (l_smps_obs < 360.0)] = (
        l_smps_obs[(l_smps_obs > 180.0) & (l_smps_obs < 360.0)] - 360.0
    )

    cond = (l_smps_obs > -100.0) & (l_smps_obs < 50.0)

    df_atnf_smps = df_atnf_smps[cond]

    # Save the properties in the SMPS dictionary.
    surveys_atnf["SMPS"]["RA"] = (
        df_atnf_smps["RAJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["DEC"] = (
        df_atnf_smps["DECJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["l_gal"] = l_smps_obs[cond]
    surveys_atnf["SMPS"]["b_gal"] = (
        df_atnf_smps["Gb"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["P"] = (
        df_atnf_smps["P0"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["P_dot"] = (
        df_atnf_smps["P1"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["DM"] = (
        df_atnf_smps["DM"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["dist"] = (
        df_atnf_smps["DIST"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["S1400"] = (
        df_atnf_smps["S1400"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["pm_RA"] = (
        df_atnf_smps["PMRA"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["pm_DEC"] = (
        df_atnf_smps["PMDEC"].to_numpy().astype(np.float64)
    )
    surveys_atnf["SMPS"]["w10"] = (
        df_atnf_smps["W10"].to_numpy().astype(np.float64)
    )

    # Merge the MeerKat TPA program data with the SMPS ATNF Pulsar Catalogue data to obtain MeerKAT flux measurements
    # for the SMPS pulsars.
    df_meerkat_smps = pd.merge(
        df_meerkat, df_atnf_smps, left_on="PSRJ", right_on="PSRJ"
    )
    df_meerkat_smps = df_meerkat_smps.dropna(subset=["ch6flux"])

    # Save the properties in the PMPS MeerKat dictionary.
    # We convert the MeerKat fluxes from [Jy] to [mJy] to compare with simulations.
    surveys_meerkat["SMPS"]["S1400"] = (
        df_meerkat_smps["ch6flux"].to_numpy().astype(np.float64) / 1000
    )
    surveys_meerkat["SMPS"]["P"] = (
        df_meerkat_smps["P0"].to_numpy().astype(np.float64)
    )
    surveys_meerkat["SMPS"]["P_dot"] = (
        df_meerkat_smps["P1"].to_numpy().astype(np.float64)
    )

    # HTRU pulsar survey database. Note that those HTRU pulsars in the ATNF Catalogue with P and Pdot
    # measurements are from the low- and mid- latitude surveys only.
    df_atnf_htru = df_atnf[df_atnf["SURVEY"].str.contains("htru_pks")]

    # Selection only pulsars falling in the HTRU sky coverage where completeness is above 90%.

    # Extracting Galactic longitude and latitude.
    b_htru_obs = df_atnf_htru["Gb"].to_numpy().astype(np.float64)
    l_htru_obs = df_atnf_htru["Gl"].to_numpy().astype(np.float64)

    # Convert galactic longitude in the range [-180., 180].
    l_htru_obs[(l_htru_obs > 180.0) & (l_htru_obs < 360.0)] = (
        l_htru_obs[(l_htru_obs > 180.0) & (l_htru_obs < 360.0)] - 360.0
    )

    cond = (
        (l_htru_obs > -120.0)
        & (l_htru_obs < 30.0)
        & (np.abs(b_htru_obs) < 15.0)
    )

    df_atnf_htru = df_atnf_htru[cond]

    # Save the properties in the HTRU dictionary.
    surveys_atnf["HTRU_low-mid"]["RA"] = (
        df_atnf_htru["RAJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["DEC"] = (
        df_atnf_htru["DECJD"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["l_gal"] = l_htru_obs[cond]
    surveys_atnf["HTRU_low-mid"]["b_gal"] = (
        df_atnf_htru["Gb"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["P"] = (
        df_atnf_htru["P0"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["P_dot"] = (
        df_atnf_htru["P1"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["DM"] = (
        df_atnf_htru["DM"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["dist"] = (
        df_atnf_htru["DIST"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["S1400"] = (
        df_atnf_htru["S1400"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["pm_RA"] = (
        df_atnf_htru["PMRA"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["pm_DEC"] = (
        df_atnf_htru["PMDEC"].to_numpy().astype(np.float64)
    )
    surveys_atnf["HTRU_low-mid"]["w10"] = (
        df_atnf_htru["W10"].to_numpy().astype(np.float64)
    )

    # Merge the MeerKat TPA program data with the HTRU ATNF Pulsar Catalogue data to obtain MeerKAT flux measurements
    # for the HTRU pulsars.
    df_meerkat_htru = pd.merge(
        df_meerkat, df_atnf_htru, left_on="PSRJ", right_on="PSRJ"
    )
    df_meerkat_htru = df_meerkat_htru.dropna(subset=["ch6flux"])

    # Save the properties in the HTRU MeerKat dictionary.
    # We convert the MeerKat fluxes from [Jy] to [mJy] to compare with simulations.
    surveys_meerkat["HTRU_low-mid"]["S1400"] = (
        df_meerkat_htru["ch6flux"].to_numpy().astype(np.float64) / 1000
    )
    surveys_meerkat["HTRU_low-mid"]["P"] = (
        df_meerkat_htru["P0"].to_numpy().astype(np.float64)
    )
    surveys_meerkat["HTRU_low-mid"]["P_dot"] = (
        df_meerkat_htru["P1"].to_numpy().astype(np.float64)
    )

    return surveys_atnf, surveys_meerkat


def load_xray_catalog(
    path_xray_catalog: pathlib.Path,
) -> dict:
    """
    Load the catalog containing the properties of observed thermally emitting X-ray bright neutron stars.

    Args:
        path_xray_catalog (pathlib.Path): Path to the catalog of thermally emitting neutron stars.

    Returns:
        (dict): A dictionary with the properties of detected neutron stars in X-rays with quiescent thermal emission.
    """

    # Initialize the dictionary where the filtered properties will be saved.
    survey_xray = {}

    # Read the thermally emitting neutron star catalog .csv file. Binary pulsars are excluded.
    df_x = pd.read_csv(
        path_xray_catalog,
        delimiter=",",
        header=[0],
    )

    # We have removed central compact objects (CCOs) because their magnetic field have likely been buried due to
    # supernova fallback and their evolution cannot be modeled with our simulation framework. We also remove neutron
    # stars associated with pulsar wind nebulae as their detection might have been triggered by the non-thermal
    # emission of the nebula. We also remove those stars that belong to the Magellanic Clouds.
    df_x = df_x[~df_x["class"].isin(["CCO"])]
    df_x = df_x[~df_x["assoc"].isin(["PWN", "PWN, Radio"])]
    df_x = df_x[~df_x["assoc"].isin(["SMC", "LMC"])]
    df_x = df_x.dropna(subset=["period(s)"])
    df_x = df_x.dropna(subset=["pdot(1e-11s/s)"])
    df_x = df_x.dropna(subset=["abs. flux (0.3-10 keV)"])

    # Extracting Galactic longitude and converting it in the range [-180., 180].
    l_x_obs = df_x["l(deg)"].to_numpy().astype(np.float64)

    l_x_obs[(l_x_obs > 180.0) & (l_x_obs < 360.0)] = (
        l_x_obs[(l_x_obs > 180.0) & (l_x_obs < 360.0)] - 360.0
    )

    # Save the properties in the X-ray dictionary.
    survey_xray["l_gal"] = df_x["l(deg)"].to_numpy().astype(np.float64)
    survey_xray["b_gal"] = df_x["b(deg)"].to_numpy().astype(np.float64)
    survey_xray["RA"] = df_x["ra(deg)"].to_numpy().astype(np.float64)
    survey_xray["DEC"] = df_x["dec(deg)"].to_numpy().astype(np.float64)
    survey_xray["dist"] = df_x["distance(kpc)"].to_numpy().astype(np.float64)
    survey_xray["P"] = df_x["period(s)"].to_numpy().astype(np.float64)
    survey_xray["P_dot"] = (
        df_x["pdot(1e-11s/s)"].to_numpy().astype(np.float64) * 1e-11
    )
    survey_xray["L_x_bol"] = (
        df_x["luminosity_bol_qui(e33erg/s)"].to_numpy().astype(np.float64)
        * 1e33
    )
    survey_xray["S_x_abs"] = (
        df_x["abs. flux (0.3-10 keV)"].to_numpy().astype(np.float64)
    )
    survey_xray["age"] = df_x["age_real(kyr)"].to_numpy().astype(np.float64)

    return survey_xray
