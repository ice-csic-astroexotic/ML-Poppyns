"""
    Generator for the observed population.

    This module creates compressed representations for the observed radio pulsar population
    in the ATNF Pulsar Catalogue using the fluxes from the TPA program on MeerKat
    presented in Posselt et al. (2023) and for the catalog of thermally emitting X-ray neutron stars.

    In the following, we work with the fluxes from the ch6flux column in Posselt et al. (2023),
    which correspond to measurements at 1429 MHz. These fluxes are used in Figure 7
    of the paper for comparison with those from the ATNF Pulsar Catalogue.

    The user can choose to generate either a dataset of images or of 2D arrays.

    More precisely we generate the following maps:

    * Four position density maps in ICRS coordinates: one for each of the three radio surveys and one for the X-ray survey
    modeled by the simulator.
    * Four $P-\\dot{P}$ density maps: one for each of the three simulated radio surveys and one for the X-ray survey.
    * Four $P-\\dot{P}$ density maps weighted by the logarithm of the radio and X-ray fluxes: three for the three simulated
    radio surveys and one for the X-ray survey.

    Display help message to run the code:

    python generate_observed_data.py --help

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

import mlpoppyns.generator.maps.dataset_density_map as dmap
import mlpoppyns.generator.maps.dataset_weighted_density_map as wdmap
from utilities.load_catalogs import (
    load_atnf_meerkat_catalog,
    load_xray_catalog,
)

log = logging.getLogger(__name__)


def create_survey_maps(
    dataset_path: str,
    survey_name: str,
    survey_dict: dict,
    survey_type: str,
    use_meerkat_fluxes: bool,
    survey_meerkat_dict: dict,
    data_type: str,
    resolution_dyn: int,
    resolution_ppdot: int,
    dictionary_density_map_radec: dict,
    dictionary_velocity_vra_map_radec: dict,
    dictionary_velocity_vdec_map_radec: dict,
    dictionary_density_map_ppdot: dict,
    dictionary_flux_map_ppdot: dict,
) -> None:
    """
    This method reads the observed population from the ATNF Pulsar Catalogue, the MeerKat TPA program (Posselt et
    al., 2023) and the thermally-emitting neutron star catalogue (Dehman et al. in prep), and subsequently generates a
    set of density maps Fin the specified format (images or arrays) and with a specified resolution.

    Args:
        dataset_path (str): Path to where the generated dataset will be saved.
        survey_name (str): Survey acronym.
        survey_dict (dict): Dictionary containing the survey data.
        survey_type (str): Survey type, radio or X-ray.
        use_meerkat_fluxes (bool): A boolean indicating whether to use MeerKat fluxes in the P-Pdot average flux maps.
        survey_meerkat_dict (dict): Dictionary containing the MeerKat survey data (used only if use_meerkat_fluxes is
            True, and survey_type == radio).
        data_type (str): Type of dataset to generate: array, array_kde, image or image_kde.
        resolution_dyn (int): Resolution (number of bins per axis for the 2D histograms) for the position and
            velocity maps to generate. In case of RA DEC maps the DEC axis has half the number of bins
            with respect to the RA axis.
        resolution_ppdot (int): Resolution (number of bins per axis for the 2D
            histograms) for the P-Pdot density maps to generate.
        dictionary_density_map_radec (dict): Dictionary containing the path to the position maps in RA, DEC for
            all the observed surveys.
        dictionary_velocity_vra_map_radec (dict): Dictionary containing the path to the proper motion maps in RA for
            all the observed surveys.
        dictionary_velocity_vdec_map_radec (dict): Dictionary containing the path to the proper motion maps in DEC for
            all the observed surveys.
        dictionary_density_map_ppdot (dict): Dictionary containing the path to the P-Pdot maps for
            all the observed surveys.
        dictionary_flux_map_ppdot (dict): Dictionary containing the path to the averaged flux P-Pdot maps for
            all the observed surveys.
    """

    # Create position density maps projected onto the RA DEC plane.
    dmap.generate_density_map(
        dataset_path,
        f"survey_{survey_name}_density_map_radec",
        0,
        data_type,
        survey_dict["ra"],
        survey_dict["dec"],
        resolution_dyn,
        int(resolution_dyn / 2),
        x_log_scale=False,
        y_log_scale=False,
        maps_dictionary=dictionary_density_map_radec,
        x_limits=(0.0, 360.0),
        y_limits=(-90.0, 90.0),
    )

    # Since the number of ATNF Catalogue objects with measured proper motions is insufficient,
    # we do not produce the velocity maps. However, for compatibility purposes with our
    # machine-learning utilities, we have to update the corresponding dictionary fields with empty strings.
    # Updating the velocity map label of component pm_RA in the RA DEC plane.
    dictionary_velocity_vra_map_radec.update(
        {f"input:survey_{survey_name}_velocity_vra_map_radec": [""]}
    )

    # Updating the velocity map label of component pm_DEC in the RA DEC plane.
    dictionary_velocity_vdec_map_radec.update(
        {f"input:survey_{survey_name}_velocity_vdec_map_radec": [""]}
    )

    # Create P-Pdot density maps.
    dmap.generate_density_map(
        dataset_path,
        f"survey_{survey_name}_density_map_ppdot",
        0,
        data_type,
        survey_dict["P"],
        survey_dict["P_dot"],
        resolution_ppdot,
        resolution_ppdot,
        x_log_scale=True,
        y_log_scale=True,
        maps_dictionary=dictionary_density_map_ppdot,
        x_limits=(0.01, 100.0),
        y_limits=(1.0e-20, 1.0e-9),
    )

    if (survey_type == "radio") & use_meerkat_fluxes:
        # Create P-Pdot average flux maps using the Meerkat fluxes.
        # To generate the average flux maps, we choose a minimum flux of 10^-7 [Jy] to assign to the empty bins since
        # all observed radio fluxes are greater than around 10^-5 [Jy].
        wdmap.generate_weighted_density_map(
            dataset_path,
            f"survey_{survey_name}_flux_map_ppdot",
            0,
            data_type,
            survey_meerkat_dict["P"],
            survey_meerkat_dict["P_dot"],
            np.log10(survey_meerkat_dict["S1400"]),
            w_min=-7,
            x_resolution=resolution_ppdot,
            y_resolution=resolution_ppdot,
            x_log_scale=True,
            y_log_scale=True,
            maps_dictionary=dictionary_flux_map_ppdot,
            x_limits=(1e-2, 1e2),
            y_limits=(1e-20, 1e-9),
        )

    elif (survey_type == "radio") & (use_meerkat_fluxes is False):
        # Create P-Pdot average flux maps using the fluxes reported in the ATNF catalog.
        # To generate the average flux maps, we choose a minimum flux of 10^-7 [Jy] to assign to the empty bins since
        # all observed radio fluxes are greater than around 10^-5 [Jy].
        wdmap.generate_weighted_density_map(
            dataset_path,
            f"survey_{survey_name}_flux_map_ppdot",
            0,
            data_type,
            survey_dict["P"],
            survey_dict["P_dot"],
            np.log10(survey_dict["S1400"]),
            w_min=-7,
            x_resolution=resolution_ppdot,
            y_resolution=resolution_ppdot,
            x_log_scale=True,
            y_log_scale=True,
            maps_dictionary=dictionary_flux_map_ppdot,
            x_limits=(1e-2, 1e2),
            y_limits=(1e-20, 1e-9),
        )

    elif survey_type == "X-ray":
        # Create P-Pdot average flux maps.
        # To generate the average flux maps, we choose a minimum flux to assign to the empty bins of 10^-17 [erg s^-1
        # cm^-2], since all observed X-ray fluxes are greater than around 10^-15 [erg s^-1 cm^-2].
        wdmap.generate_weighted_density_map(
            dataset_path,
            f"survey_{survey_name}_flux_map_ppdot",
            0,
            data_type,
            survey_dict["P"],
            survey_dict["P_dot"],
            np.log10(survey_dict["S_x_abs"]),
            w_min=-17,
            x_resolution=resolution_ppdot,
            y_resolution=resolution_ppdot,
            x_log_scale=True,
            y_log_scale=True,
            maps_dictionary=dictionary_flux_map_ppdot,
            x_limits=(1e-2, 1e2),
            y_limits=(1e-20, 1e-9),
        )

    else:
        log.error(
            f"The specified {survey_type} is not supported. Choose between 'radio' and 'X-ray'."
        )


def generate_dataset(args: argparse.Namespace) -> None:
    """
    This method generates a dataset of density maps in the specified format (images or arrays) and with a specified
    resolution from the ATNF Pulsar Catalogue, the MeerKat TPA program (Posselt et al., 2023) and the thermally-emitting
    neutron star catalogue (Dehman et al. in prep).
    All the information about the dataset is stored in a dataset.csv file containing the density-map file names
    and the set of parameter values for each simulated population.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the following attributes:

            - path_atnf (str): Path to the ATNF catalogue.
            - path_meerkat (str): Path to the MeerKat catalogue.
            - path_xray (str): Path to the thermally-emitting neutron star catalogue.
            - save_dir (str): Path to where the generated dataset will be saved.
            - data_type (str): Type of dataset to generate: array, array_kde, image or image_kde.
            - resolution_ppdot_radio (int): Resolution (number of bins per axis for the 2D
                histograms) for the P-Pdot maps related to the radio surveys.
            - resolution_dyn_radio (int): Resolution (number of bins per axis for the 2D
                histograms) for the dynamical maps related to the radio surveys. In case of RA DEC maps the
                DEC axis has half the number of bins with respect to the RA axis.
            - resolution_ppdot_xray (int): Resolution (number of bins per axis for the 2D
                histograms) for the P-Pdot maps related to the X-ray survey.
            - resolution_dyn_xray (int): Resolution (number of bins per axis for the 2D
                histograms) for the dynamical maps related to the X-ray surveys. In case of RA DEC maps the
                DEC axis has half the number of bins with respect to the RA axis.
            - filter_young_xdins (bool): Whether to filter the X-ray simulated samples to include only young magnetars
                and XDINS-like sources (default is False).
    """

    catalog_atnf, catalog_meerkat = load_atnf_meerkat_catalog(
        args.path_atnf, args.path_meerkat
    )
    catalog_xray = load_xray_catalog(args.path_xray, args.filter_young_xdins)

    # Create the dataset directory path.
    dataset_path = f"{args.save_dir}"
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)

    # Initialize dictionaries that will contain the density-map file names and
    # the corresponding set of parameter values.
    survey_PMPS_density_map_radec_dictionary = {}
    survey_PMPS_velocity_vra_map_radec_dictionary = {}
    survey_PMPS_velocity_vdec_map_radec_dictionary = {}
    survey_PMPS_density_map_ppdot_dictionary = {}
    survey_PMPS_flux_map_ppdot_dictionary = {}

    survey_SMPS_density_map_radec_dictionary = {}
    survey_SMPS_velocity_vra_map_radec_dictionary = {}
    survey_SMPS_velocity_vdec_map_radec_dictionary = {}
    survey_SMPS_density_map_ppdot_dictionary = {}
    survey_SMPS_flux_map_ppdot_dictionary = {}

    survey_HTRU_density_map_radec_dictionary = {}
    survey_HTRU_velocity_vra_map_radec_dictionary = {}
    survey_HTRU_velocity_vdec_map_radec_dictionary = {}
    survey_HTRU_density_map_ppdot_dictionary = {}
    survey_HTRU_flux_map_ppdot_dictionary = {}

    survey_xray_density_map_radec_dictionary = {}
    survey_xray_velocity_vra_map_radec_dictionary = {}
    survey_xray_velocity_vdec_map_radec_dictionary = {}
    survey_xray_density_map_ppdot_dictionary = {}
    survey_xray_flux_map_ppdot_dictionary = {}

    log.info("Generating sample...")

    # Create a set of maps for each survey.
    create_survey_maps(
        dataset_path,
        "PMPS",
        catalog_atnf["PMPS"],
        "radio",
        True,
        catalog_meerkat["PMPS"],
        args.data_type,
        args.resolution_dyn_radio,
        args.resolution_ppdot_radio,
        survey_PMPS_density_map_radec_dictionary,
        survey_PMPS_velocity_vra_map_radec_dictionary,
        survey_PMPS_velocity_vdec_map_radec_dictionary,
        survey_PMPS_density_map_ppdot_dictionary,
        survey_PMPS_flux_map_ppdot_dictionary,
    )

    create_survey_maps(
        dataset_path,
        "SMPS",
        catalog_atnf["SMPS"],
        "radio",
        True,
        catalog_meerkat["SMPS"],
        args.data_type,
        args.resolution_dyn_radio,
        args.resolution_ppdot_radio,
        survey_SMPS_density_map_radec_dictionary,
        survey_SMPS_velocity_vra_map_radec_dictionary,
        survey_SMPS_velocity_vdec_map_radec_dictionary,
        survey_SMPS_density_map_ppdot_dictionary,
        survey_SMPS_flux_map_ppdot_dictionary,
    )

    create_survey_maps(
        dataset_path,
        "HTRU",
        catalog_atnf["HTRU_low-mid"],
        "radio",
        True,
        catalog_meerkat["HTRU_low-mid"],
        args.data_type,
        args.resolution_dyn_radio,
        args.resolution_ppdot_radio,
        survey_HTRU_density_map_radec_dictionary,
        survey_HTRU_velocity_vra_map_radec_dictionary,
        survey_HTRU_velocity_vdec_map_radec_dictionary,
        survey_HTRU_density_map_ppdot_dictionary,
        survey_HTRU_flux_map_ppdot_dictionary,
    )

    create_survey_maps(
        dataset_path,
        "xray",
        catalog_xray,
        "X-ray",
        False,
        catalog_meerkat["HTRU_low-mid"],
        args.data_type,
        args.resolution_dyn_xray,
        args.resolution_ppdot_xray,
        survey_xray_density_map_radec_dictionary,
        survey_xray_velocity_vra_map_radec_dictionary,
        survey_xray_velocity_vdec_map_radec_dictionary,
        survey_xray_density_map_ppdot_dictionary,
        survey_xray_flux_map_ppdot_dictionary,
    )

    # Merge the filename and parameter dictionaries into a single dictionary.
    dataset_dictionary = {
        **survey_PMPS_density_map_radec_dictionary,
        **survey_SMPS_density_map_radec_dictionary,
        **survey_HTRU_density_map_radec_dictionary,
        **survey_xray_density_map_radec_dictionary,
        **survey_PMPS_velocity_vra_map_radec_dictionary,
        **survey_SMPS_velocity_vra_map_radec_dictionary,
        **survey_HTRU_velocity_vra_map_radec_dictionary,
        **survey_xray_velocity_vra_map_radec_dictionary,
        **survey_PMPS_velocity_vdec_map_radec_dictionary,
        **survey_SMPS_velocity_vdec_map_radec_dictionary,
        **survey_HTRU_velocity_vdec_map_radec_dictionary,
        **survey_xray_velocity_vdec_map_radec_dictionary,
        **survey_PMPS_density_map_ppdot_dictionary,
        **survey_SMPS_density_map_ppdot_dictionary,
        **survey_HTRU_density_map_ppdot_dictionary,
        **survey_xray_density_map_ppdot_dictionary,
        **survey_PMPS_flux_map_ppdot_dictionary,
        **survey_SMPS_flux_map_ppdot_dictionary,
        **survey_HTRU_flux_map_ppdot_dictionary,
        **survey_xray_flux_map_ppdot_dictionary,
    }

    # Write the whole dataset dictionary into a .csv file.
    dataset_filename = f"{dataset_path}/dataset_observed.csv"

    df = pd.DataFrame(
        {key: pd.Series(value) for key, value in dataset_dictionary.items()}
    )
    df.to_csv(dataset_filename, encoding="utf-8", index=False)

    log.info("File dataset_observed.csv generated")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "--path_atnf",
        nargs="?",
        type=str,
        default="data/observations/atnf_full_nobinary_24-09-2024_with_errors.csv",
        help="Path, with the name of the csv included, to where the ATNF Pulsar Catalogue is located.",
    )
    parser.add_argument(
        "--path_meerkat",
        nargs="?",
        type=str,
        default="data/observations/meerkat_tpa_posselt_2023.csv",
        help="Path, with the name of the csv included, to where the MeerKat TPA program data is located.",
    )
    parser.add_argument(
        "--path_xray",
        nargs="?",
        type=str,
        default="data/observations/thermal_NS_05-11-2024.csv",
        help="Path, with the name of the csv included, to where the thermally emitting neutron star data is located.",
    )
    parser.add_argument(
        "--save_dir",
        nargs="?",
        type=str,
        default="output/gen_obs",
        help="Path to the folder, where the dataset will be saved.",
    )
    parser.add_argument(
        "--data_type",
        nargs="?",
        type=str,
        choices=["array", "array_kde", "image", "image_kde"],
        default="array",
        help="Type of dataset to generate: array, array_kde, or image, image_kde.",
    )
    parser.add_argument(
        "--resolution_dyn_radio",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the position and velocity maps for the radio surveys that will be generated (in number of bins).",
    )
    parser.add_argument(
        "--resolution_ppdot_radio",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the P-Pdot maps for the radio surveys that will be generated (in number of bins).",
    )
    parser.add_argument(
        "--resolution_dyn_xray",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the position and velocity maps for the X-ray surveys that will be generated (in number of bins).",
    )
    parser.add_argument(
        "--resolution_ppdot_xray",
        nargs="?",
        type=int,
        default=64,
        help="Resolution of the P-Pdot maps for the X-ray surveys that will be generated (in number of bins).",
    )
    parser.add_argument(
        "--filter_young_xdins",
        action="store_true",
        help="Whether to filter the X-ray simulated samples to include only young magnetars and XDINS-like sources (default is False).",
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    generate_dataset(args)
