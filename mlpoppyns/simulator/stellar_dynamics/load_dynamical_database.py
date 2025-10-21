"""
    Module to load a dynamically evolved population database.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)
        Alberto Garcia-Garcia (garciagarcia @ ice.csic.es)
        Celsa Pardo Araujo (pardo @ ice.csic.es)
"""

import json
import logging
import pathlib
import sys

import mlpoppyns.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.samplers.memory_efficient_sampling as mes
from mlpoppyns.simulator.config_simulator import cfg


def load_database_dyn(
    dyn_path: pathlib.Path,
    n_batchsize: int,
    idx_remove: list,
    logger: logging.Logger,
) -> dict:
    """
    Load a batch of a dynamically evolved population from a .csv file, convert coordinates and return a dictionary
    with the dynamical information of the selected stars.

    Args:
        dyn_path (pathlib.Path): Path to the directory containing the dynamically evolved population data.
        n_batchsize (int): Batch size of stars to select when loading the data.
        idx_remove (list): List of indices to remove from the dynamical database.
        logger (logging.Logger): Logger to use.

    Returns:
        (dict): A dictionary containing the data of the selected dynamical population chunk.
    """

    # Check if the parsed dynamically simulated population directory exists.
    dyn_path = pathlib.Path(dyn_path)
    dyn_config_path = dyn_path / "configuration.json"
    dyn_data_path = dyn_path / "final_pop_dyn.csv"

    if not dyn_data_path.exists():
        logger.error(f"File {dyn_data_path} not found...")
        sys.exit()

    with open(dyn_config_path, "r") as f:
        config_dyn = json.load(f)

    # Load the batch of the file containing the dynamically evolved population parameters.
    df_dyn = mes.select(
        dyn_data_path,
        n_batchsize,
        config_dyn["NS_number"],
        idx_remove,
    )

    # Remove the second line of the header and convert the dataframe to a dictionary.
    df_dyn.columns = df_dyn.columns.get_level_values(0)
    dyn_dict = {col: df_dyn[col].to_numpy() for col in df_dyn.columns}

    # Add the dynamical properties in all coordinates and the index associated to each star and remove unnecessary keys.
    dictionary_dyn_database_chunk = (
        coco.convert_cylindrical_to_all_coordinates(dyn_dict)
    )
    dictionary_dyn_database_chunk["idx"] = df_dyn.index.values
    dictionary_dyn_database_chunk.pop("r", None)
    dictionary_dyn_database_chunk.pop("phi", None)
    dictionary_dyn_database_chunk.pop("z", None)
    dictionary_dyn_database_chunk.pop("v_r", None)
    dictionary_dyn_database_chunk.pop("v_phi", None)
    dictionary_dyn_database_chunk.pop("v_z", None)

    # Update the parameters in the simulation configuration file with the ones of the dynamical database.
    cfg["t_age_max"] = config_dyn["t_age_max"]
    cfg["NS_number"] = config_dyn["NS_number"]
    cfg["kick_model"] = config_dyn["kick_model"]
    if config_dyn["kick_model"] == "km_maxwell":
        cfg["sigma_k"] = config_dyn["sigma_k"]
    elif config_dyn["kick_model"] == "km_exp":
        cfg["vk_c"] = config_dyn["vk_c"]
    elif config_dyn["kick_model"] == "km_double_maxwell":
        cfg["sigma_k_comp1"] = config_dyn["sigma_k_comp1"]
        cfg["sigma_k_comp2"] = config_dyn["sigma_k_comp2"]
        cfg["kick_weight_comp1"] = config_dyn["kick_weight_comp1"]
    cfg["h_c"] = config_dyn["h_c"]

    # Add the path of the dynamical database in the configuration file.
    cfg["dyn_database_path"] = str(dyn_path)

    return dictionary_dyn_database_chunk
