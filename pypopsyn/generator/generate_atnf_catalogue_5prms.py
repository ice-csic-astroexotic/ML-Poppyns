"""
    Generator for the observed population.

    This module creates compressed representations for the observed population in the ATNF Pulsar Catalogue.

    The user can choose to generate either a dataset of images or of 2D arrays.

    Display help message to run the code:

    python generate_atnf_catalogue.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
        Celsa Pardo Araujo  (pardo@ice.csic.es)
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

import pypopsyn.generator.maps.position_maps as pmaps
import pypopsyn.generator.maps.ppdot_maps as ppdmaps

log = logging.getLogger(__name__)


def create_survey_maps(
    dataset_path: str,
    survey_name: str,
    data_type: str,
    resolution_dyn: int,
    resolution_ppdot: int,
    dictionary_position_map_radec: dict,
    dictionary_velocity_map_vra: dict,
    dictionary_velocity_map_vdec: dict,
    dictionary_ppdot_map: dict,
    P: np.ndarray,
    Pdot: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> None:
    """
    This method reads the observed population from the ATNF Pulsar Catalogue
    and generates a set of density maps in the specified format (images or arrays) and with a specified resolution.

    Args:

        dataset_path (str): Path to where the generated dataset will be saved.

        survey_name (str): Survey acronym.

        data_type (str): Type of dataset to generate: array or image.

        resolution_dyn (int): Resolution (number of bins per axis for the 2d histograms) for the position and
            velocity maps to generate. In case of RA DEC maps the DEC axis has half the number of bins
            with respect to the RA axis.

        resolution_ppdot (int): Resolution (number of bins per axis for the 2d
            histograms) for the P-Pdot density maps to generate.

        dictionary_position_map_radec (dict): Dictionary containing the path to the position maps in RA, DEC for
            all the simulated surveys.

        dictionary_velocity_map_vra (dict): Dictionary containing the path to the proper motion maps in RA for
            all the simulated surveys.

        dictionary_velocity_map_vdec (dict): Dictionary containing the path to the proper motion maps in DEC for
            all the simulated surveys.

        dictionary_ppdot_map (dict): Dictionary containing the path to the P-Pdot maps for
            all the simulated surveys.

        P (np.ndarray): Array of spin periods of the pulsars in [s].

        Pdot (np.ndarray): Array of spin period derivatives of the pulsars in [s/s].

        ra (np.ndarray): Right ascension in [deg] defined between [0, 360] deg in ICRS frame.

        dec (np.ndarray): Declination in [deg] defined between [-90, 90] deg in ICRS frame.

    Returns:
        Nothing.
    """

    # Create position density maps projected onto the RA DEC plane.
    pmaps.generate_position_map(
        dataset_path,
        f"survey_{survey_name}_position_map_radec",
        0,
        data_type,
        ra,
        dec,
        resolution_dyn,
        int(args.resolution_dyn / 2),
        dictionary_position_map_radec,
        x_limits=(0.0, 360.0),
        y_limits=(-90.0, 90.0),
    )

    # Since the number of ATNF Catalogue objects with measured proper motions is insufficient,
    # we do not produce the velocity maps. However, for compatibility purposes with our
    # machine-learning utilities, we have to update the corresponding dictionary fields with empty strings.
    # Updating the velocity map label of component v_RA in the RA DEC plane.
    dictionary_velocity_map_vra.update(
        {f"input:survey_{survey_name}_velocity_map_vra": [""]}
    )

    # Updating the velocity map label of component v_DEC in the RA DEC plane.
    dictionary_velocity_map_vdec.update(
        {f"input:survey_{survey_name}_velocity_map_vdec": [""]}
    )

    # Create P-Pdot density maps.
    ppdmaps.generate_ppdot_map(
        dataset_path,
        f"survey_{survey_name}_ppdot_map",
        0,
        data_type,
        P,
        Pdot,
        resolution_ppdot,
        resolution_ppdot,
        dictionary_ppdot_map,
    )


def generate_dataset(args) -> None:
    """
    This method generates a dataset of density maps in the specified format (images or arrays) and with a specified
    resolution from the ATNF Pulsar Catalogue.
    All the information about the dataset is stored in a dataset.csv file containing the density-map file names
    and the set of parameter values for each simulated population.

    Args:
        args:
            data (str): Path to where the observed population is located.

            save_dir (str): Path to where the generated dataset will be saved.

            data_type (str): Type of dataset to generate: array or image.

            resolution_ppdot (int): Resolution (number of bins per axis for the 2d
            histograms) for the P-Pdot density maps to generate.

    Returns:
        Nothing.
    """

    # Create the dataset directory path.
    dataset_path = f"{args.save_dir}"
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # Initialize dictionaries that will contain the density-map file names and
    # the corresponding set of parameter values.
    survey_PMPS_position_map_radec_dictionary = {}
    survey_PMPS_velocity_map_vra_dictionary = {}
    survey_PMPS_velocity_map_vdec_dictionary = {}
    survey_PMPS_ppdot_map_dictionary = {}

    survey_SMPS_position_map_radec_dictionary = {}
    survey_SMPS_velocity_map_vra_dictionary = {}
    survey_SMPS_velocity_map_vdec_dictionary = {}
    survey_SMPS_ppdot_map_dictionary = {}

    survey_HTRU_position_map_radec_dictionary = {}
    survey_HTRU_velocity_map_vra_dictionary = {}
    survey_HTRU_velocity_map_vdec_dictionary = {}
    survey_HTRU_ppdot_map_dictionary = {}

    param_dictionary = {}

    # Read the full ATNF catalogue.csv file. Binary pulsars are excluded.
    df_atnf = pd.read_csv(
        args.data,
        delimiter=";",
        header=[0, 1],
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

    df_atnf = df_atnf[~df_atnf["P0"]["(s)"].isin(["NAN"])]
    df_atnf = df_atnf[~df_atnf["P1"]["(s/s)"].isin(["NAN"])]
    df_atnf = df_atnf[
        ~df_atnf["ASSOC"]["Unnamed: 24_level_1"].str.match("|".join(discard))
    ]

    # Select only isolated, non-recycled neutron stars, i.e., those with Pdot > 1e-19.
    df_atnf = df_atnf[
        df_atnf["P1"]["(s/s)"].to_numpy().astype(np.float64) > 1.0e-19
    ]

    # Parkes multibeam pulsar survey database.
    df_atnf_pmps = df_atnf[
        df_atnf["SURVEY"]["Unnamed: 25_level_1"].str.contains("pksmb")
    ]

    # Extracting Galactic longitude, latitude, period and period derivative.
    RA_pmps_obs = df_atnf_pmps["RAJD"]["(deg)"].to_numpy().astype(np.float64)
    DEC_pmps_obs = df_atnf_pmps["DECJD"]["(deg)"].to_numpy().astype(np.float64)
    l_pmps_obs = df_atnf_pmps["Gl"]["(deg)"].to_numpy().astype(np.float64)
    b_pmps_obs = df_atnf_pmps["Gb"]["(deg)"].to_numpy().astype(np.float64)
    P_pmps_obs = df_atnf_pmps["P0"]["(s)"].to_numpy().astype(np.float64)
    Pdot_pmps_obs = df_atnf_pmps["P1"]["(s/s)"].to_numpy().astype(np.float64)

    # Converting galactic latitude into the range [-180., 180].
    l_pmps_obs[(l_pmps_obs > 180.0) & (l_pmps_obs < 360.0)] = (
        l_pmps_obs[(l_pmps_obs > 180.0) & (l_pmps_obs < 360.0)] - 360.0
    )

    # Select only pulsars falling into the Parkes multibeam sky coverage where completeness is above 90%.
    # See Lorimer et al. (2006) for details.
    cond = (
        (l_pmps_obs > -100.0)
        & (l_pmps_obs < 50.0)
        & (np.abs(b_pmps_obs) < 5.0)
    )

    RA_pmps_obs = RA_pmps_obs[cond]
    DEC_pmps_obs = DEC_pmps_obs[cond]
    P_pmps_obs = P_pmps_obs[cond]
    Pdot_pmps_obs = Pdot_pmps_obs[cond]

    # Swinburne multibeam pulsar survey database.
    df_atnf_smps = df_atnf[
        df_atnf["SURVEY"]["Unnamed: 25_level_1"].str.contains("pkssw")
    ]

    # Extracting Galactic longitude, latitude, right ascension and declination, period and period derivative.
    RA_smps_obs = df_atnf_smps["RAJD"]["(deg)"].to_numpy().astype(np.float64)
    DEC_smps_obs = df_atnf_smps["DECJD"]["(deg)"].to_numpy().astype(np.float64)
    l_smps_obs = df_atnf_smps["Gl"]["(deg)"].to_numpy().astype(np.float64)
    P_smps_obs = df_atnf_smps["P0"]["(s)"].to_numpy().astype(np.float64)
    Pdot_smps_obs = df_atnf_smps["P1"]["(s/s)"].to_numpy().astype(np.float64)

    # Converting galactic latitude into the range [-180., 180].
    l_smps_obs[(l_smps_obs > 180.0) & (l_smps_obs < 360.0)] = (
        l_smps_obs[(l_smps_obs > 180.0) & (l_smps_obs < 360.0)] - 360.0
    )

    # Select only pulsars falling into the Swinburne sky coverage where completeness is above 90%.
    # See Edwards et al. (2001) and Jacoby et al. (2009) for details.
    cond = (l_smps_obs > -100.0) & (l_smps_obs < 50.0)

    RA_smps_obs = RA_smps_obs[cond]
    DEC_smps_obs = DEC_smps_obs[cond]
    P_smps_obs = P_smps_obs[cond]
    Pdot_smps_obs = Pdot_smps_obs[cond]

    # HTRU pulsar survey database. Note that those HTRU pulsars in the ATNF Catalogue with P and Pdot
    # measurements are from the low- and mid- latitude surveys only.
    df_atnf_htru = df_atnf[
        df_atnf["SURVEY"]["Unnamed: 25_level_1"].str.contains("htru_pks")
    ]

    RA_htru_obs = df_atnf_htru["RAJD"]["(deg)"].to_numpy().astype(np.float64)
    DEC_htru_obs = df_atnf_htru["DECJD"]["(deg)"].to_numpy().astype(np.float64)
    P_htru_obs = df_atnf_htru["P0"]["(s)"].to_numpy().astype(np.float64)
    Pdot_htru_obs = df_atnf_htru["P1"]["(s/s)"].to_numpy().astype(np.float64)

    log.info("Generating sample...")

    # Create a set of maps for each survey.
    create_survey_maps(
        dataset_path,
        "PMPS",
        args.data_type,
        args.resolution_dyn,
        args.resolution_ppdot,
        survey_PMPS_position_map_radec_dictionary,
        survey_PMPS_velocity_map_vra_dictionary,
        survey_PMPS_velocity_map_vdec_dictionary,
        survey_PMPS_ppdot_map_dictionary,
        P_pmps_obs,
        Pdot_pmps_obs,
        RA_pmps_obs,
        DEC_pmps_obs,
    )

    create_survey_maps(
        dataset_path,
        "SMPS",
        args.data_type,
        args.resolution_dyn,
        args.resolution_ppdot,
        survey_SMPS_position_map_radec_dictionary,
        survey_SMPS_velocity_map_vra_dictionary,
        survey_SMPS_velocity_map_vdec_dictionary,
        survey_SMPS_ppdot_map_dictionary,
        P_smps_obs,
        Pdot_smps_obs,
        RA_smps_obs,
        DEC_smps_obs,
    )

    create_survey_maps(
        dataset_path,
        "HTRU",
        args.data_type,
        args.resolution_dyn,
        args.resolution_ppdot,
        survey_HTRU_position_map_radec_dictionary,
        survey_HTRU_velocity_map_vra_dictionary,
        survey_HTRU_velocity_map_vdec_dictionary,
        survey_HTRU_ppdot_map_dictionary,
        P_htru_obs,
        Pdot_htru_obs,
        RA_htru_obs,
        DEC_htru_obs,
    )

    # Save the initial simulation parameter values as a dictionary.
    # Because we do not know these for the observed values, we set them to NaN.
    param_dictionary.update(
        {
            "B_initial_log10_mean": [np.nan],
            "B_initial_log10_sigma": [np.nan],
            "P_initial_log10_mean": [np.nan],
            "P_initial_log10_sigma": [np.nan],
            "a_late": [np.nan],
            "h_c": [np.nan],
            "sigma_k": [np.nan],
        }
    )

    # Merge the filename and parameter dictionaries into a single dictionary.
    dataset_dictionary = {
        **survey_PMPS_position_map_radec_dictionary,
        **survey_SMPS_position_map_radec_dictionary,
        **survey_HTRU_position_map_radec_dictionary,
        **survey_PMPS_velocity_map_vra_dictionary,
        **survey_SMPS_velocity_map_vra_dictionary,
        **survey_HTRU_velocity_map_vra_dictionary,
        **survey_PMPS_velocity_map_vdec_dictionary,
        **survey_SMPS_velocity_map_vdec_dictionary,
        **survey_HTRU_velocity_map_vdec_dictionary,
        **survey_PMPS_ppdot_map_dictionary,
        **survey_SMPS_ppdot_map_dictionary,
        **survey_HTRU_ppdot_map_dictionary,
        **param_dictionary,
    }

    # Write the whole dataset dictionary into a .csv file.
    dataset_filename = f"{dataset_path}/dataset_atnf.csv"

    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)

    log.info("File dataset_atnf.csv generated")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--data",
        nargs="?",
        type=str,
        default="data/observations/atnf_full_nobinary_25-04-2023.csv",
        help="Path, with the name of the csv included, to where the ATNF Pulsar Catalogue is located.",
    )
    parser.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="output/gen_atnf",
        help="Path to the folder, where the dataset will be saved.",
    )
    parser.add_argument(
        "--data_type",
        nargs="?",
        type=str,
        choices=["array", "image"],
        default="array",
        help="Type of dataset to generate: array or image.",
    )
    parser.add_argument(
        "--resolution_dyn",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the position and velocity maps that will be generated (in number of bins).",
    )
    parser.add_argument(
        "--resolution_ppdot",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the P-Pdot maps that will be generated (in number of bins).",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
