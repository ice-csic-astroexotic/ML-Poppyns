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

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import utilities.samplers.memory_efficient_sampling as mes
from pypopsyn.simulator.config_simulator import cfg


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

    age = df_dyn["age"]["[yr]"].to_numpy()
    r = df_dyn["r"]["[kpc]"].to_numpy()
    phi = df_dyn["phi"]["[rad]"].to_numpy()
    z = df_dyn["z"]["[kpc]"].to_numpy()
    v_r = df_dyn["v_r"]["[km/s]"].to_numpy()
    v_phi = df_dyn["v_phi"]["[km/s]"].to_numpy()
    v_z = df_dyn["v_z"]["[km/s]"].to_numpy()

    # Convert from polar coordinates to Cartesian coordinates.
    x, y = coco.polar_to_cartesian(r, phi)

    # Convert velocity components from galactocentric cylindrical coordinates
    # to galactocentric Cartesian coordinates.
    (
        v_x,
        v_y,
        v_z,
    ) = coco.speed_cylindrical_to_cartesian(v_r, v_phi, v_z, phi)

    # Convert galactocentric coordinates and velocities into ICRS frame.
    (
        ra,
        dec,
        dist_heliocentric_icrs,
        pm_ra,
        pm_dec,
        v_ls_icrs,
    ) = coco.galactocentric_to_icrs(x, y, z, v_x, v_y, v_z)

    # Convert galactocentric coordinates and velocities into galactic coordinates.
    (
        l_gal,
        b_gal,
        dist_heliocentric_gal,
        pm_l,
        pm_b,
        v_ls_gal,
    ) = coco.galactocentric_to_galactic(x, y, z, v_x, v_y, v_z)

    dictionary_dyn_database_chunk = {
        "age": age,
        "ra": ra,
        "dec": dec,
        "l": l_gal,
        "b": b_gal,
        "dist": dist_heliocentric_icrs,
        "pm_ra": pm_ra,
        "pm_dec": pm_dec,
        "v_ls": v_ls_icrs,
        "idx": df_dyn.index.values,
    }

    # Update the parameters in the simulation configuration file with the ones of the dynamical database.
    cfg["t_age_max"] = config_dyn["t_age_max"]
    cfg["NS_number"] = config_dyn["NS_number"]
    cfg["kick_model"] = config_dyn["kick_model"]
    cfg["sigma_k"] = config_dyn["sigma_k"]
    cfg["vk_c"] = config_dyn["vk_c"]
    cfg["h_c"] = config_dyn["h_c"]

    # Add the path of the dynamical database in the configuration file.
    cfg["dyn_database_path"] = str(dyn_path)

    return dictionary_dyn_database_chunk
