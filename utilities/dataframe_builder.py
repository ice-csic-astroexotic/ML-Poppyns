"""
    Helper function to create a DataFrame with a MultiIndex header.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)
"""


import pandas as pd


def build_dataframe(
    data_dict: dict, parameters: list, units: list
) -> pd.DataFrame:
    """
    Helper function to create a DataFrame with a MultiIndex header.

    Args:
        data_dict (dict): A dictionary containing the data to be saved in the DataFrame.
        parameters (list): A list of parameter names, used as the first level of the MultiIndex header.
        units (list): A list of physical units, used as the second level of the MultiIndex header.

    Returns:
        (pd.DataFrame): A Pandas DataFrame with a MultiIndex header, where columns are
            indexed by parameters and units.
    """

    header = pd.MultiIndex.from_arrays([parameters, units])

    # Force column order to match the parameters.
    df = pd.DataFrame({key: data_dict[key] for key in parameters})
    df.columns = header

    return df


def create_output_dataframe_dyn(
    dictionary_final_pop_dyn: dict,
) -> pd.DataFrame:
    """
    Creating Pandas DataFrames containing the properties of the dynamically evolved neutron stars.

    Args:
        dictionary_final_pop_dyn (dict): Dictionary containing evolved neutron star properties.

    Returns:
        (pd.DataFrame): A DataFrame containing the neutron stars' properties.
    """

    # Generating two header lines and merging them using MultiIndex.
    parameters = [
        "age",
        "r",
        "phi",
        "z",
        "v_r",
        "v_phi",
        "v_z",
    ]
    units = [
        "[yr]",
        "[kpc]",
        "[rad]",
        "[kpc]",
        "[km/s]",
        "[km/s]",
        "[km/s]",
    ]

    # Build the DataFrame using the appropriate parameters and units.
    df = build_dataframe(dictionary_final_pop_dyn, parameters, units)

    return df


def create_output_dataframe_initial_pop(
    dictionary_initial_pop: dict,
) -> pd.DataFrame:
    """
    Creates Pandas DataFrames containing the birth properties of the neutron stars for the full simulation.

    Args:
        dictionary_initial_pop (dict): Dictionary containing neutron star properties at birth.

    Returns:
        (pd.DataFrame): A DataFrame containing the neutron stars' properties.
    """

    # Generating two header lines and merging them using MultiIndex.
    parameters = [
        "idx",
        "age",
        "r",
        "phi",
        "z",
        "v_r",
        "v_phi",
        "v_z",
        "v_orb",
        "B",
        "chi",
        "P",
        "P_dot",
    ]
    units = [
        " ",
        "[yr]",
        "[kpc]",
        "[rad]",
        "[kpc]",
        "[kpc yr^-1]",
        "[kpc yr^-1]",
        "[kpc yr^-1]",
        "[kpc yr^-1]",
        "[G]",
        "[rad]",
        "[s]",
        "[s s^-1]",
    ]

    # Build the DataFrame using the appropriate parameters and units.
    df = build_dataframe(dictionary_initial_pop, parameters, units)

    return df


def create_output_dataframe_final_pop(
    dictionary_final_pop: dict,
    simulation_xray: bool,
) -> pd.DataFrame:
    """
    Creating Pandas DataFrames containing the properties of the evolved neutron stars for the full simulation.

    Args:
        dictionary_final_pop (dict): Dictionary containing evolved neutron star properties.
        simulation_xray (bool): If True, the dataframe contains also the parameters of an X-ray simulation.

    Returns:
        (pd.DataFrame): A DataFrame containing the neutron stars' properties.
    """

    if simulation_xray:
        # Generating two header lines and merging them using MultiIndex.
        parameters = [
            "idx",
            "age",
            "r",
            "phi",
            "z",
            "ra",
            "dec",
            "l",
            "b",
            "dist",
            "v_r",
            "v_phi",
            "v_z",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "B",
            "B_initial",
            "chi",
            "P",
            "P_dot",
            "L_radio_bol",
            "S_radio_bol",
            "w_int",
            "DM",
            "tau_sc",
            "intercepted_radio",
            "spectral_index",
            "L_x_therm",
            "S_x_rcs_abs",
            "S_x_bb_abs",
            "N_H",
            "outburst",
        ]
        units = [
            " ",
            "[yr]",
            "[kpc]",
            "[rad]",
            "[kpc]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[kpc]",
            "[km s^-1]",
            "[km s^-1]",
            "[km s^-1]",
            "[mas yr^-1]",
            "[mas yr^-1]",
            "[km s^-1]",
            "[G]",
            "[G]",
            "[rad]",
            "[s]",
            "[s s^-1]",
            "[erg s^-1]",
            "[erg s^-1 cm^-2]",
            "[s]",
            "[pc cm^-3]",
            "[s]",
            " ",
            " ",
            "[erg s^-1]",
            "[erg s^-1 cm^-2]",
            "[erg s^-1 cm^-2]",
            "[cm^-2]",
            " ",
        ]

    else:
        # Generating two header lines and merging them using MultiIndex.
        parameters = [
            "idx",
            "age",
            "r",
            "phi",
            "z",
            "ra",
            "dec",
            "l",
            "b",
            "dist",
            "v_r",
            "v_phi",
            "v_z",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "B",
            "B_initial",
            "chi",
            "P",
            "P_dot",
            "L_radio_bol",
            "S_radio_bol",
            "w_int",
            "DM",
            "tau_sc",
            "intercepted_radio",
            "spectral_index",
        ]
        units = [
            " ",
            "[yr]",
            "[kpc]",
            "[rad]",
            "[kpc]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[deg]",
            "[kpc]",
            "[km s^-1]",
            "[km s^-1]",
            "[km s^-1]",
            "[mas yr^-1]",
            "[mas yr^-1]",
            "[km s^-1]",
            "[G]",
            "[G]",
            "[rad]",
            "[s]",
            "[s s^-1]",
            "[erg s^-1]",
            "[erg s^-1 cm^-2]",
            "[s]",
            "[pc cm^-3]",
            "[s]",
            " ",
            " ",
        ]

    # Build the DataFrame using the appropriate parameters and units.
    df = build_dataframe(dictionary_final_pop, parameters, units)

    return df
