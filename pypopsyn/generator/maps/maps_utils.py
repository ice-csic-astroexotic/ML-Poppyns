"""
    Adjusting input data and bin edges for a specific scaling of the axes.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
"""

import typing

import numpy as np


def check_range_above_zero(r_range: typing.Tuple[float, float]) -> None:
    """
    Check that a range larger than zero is provided.

    Args:
        r_range (Tuple[float, float]): Range of values for the r coordinate to be checked.
    """

    if (r_range[0] <= 0) or (r_range[1] <= 0):
        raise ValueError("Value range has to be above zero.")


def remove_nan_entries(*arrays: np.ndarray) -> typing.Tuple[np.ndarray, ...]:
    """
    Removes entries at positions where any of the input arrays contains a NaN.

    This function accepts any number of 1D NumPy arrays of equal length and returns
    new arrays (as a tuple) with the same indices removed across all inputs where
    at least one array had a NaN.

    Args:
        *arrays (np.ndarray): Two or more 1D NumPy arrays of the same length.

    Returns:
        Tuple[np.ndarray, ...]: A tuple containing cleaned NumPy arrays, where
            all arrays have the same length and NaN-containing positions have been removed.
    """
    if len(arrays) == 0:
        return tuple()

    length = len(arrays[0])
    if any(len(arr) != length for arr in arrays):
        raise ValueError("All input arrays must have the same length.")

    stacked = np.vstack(arrays)
    mask = ~np.any(np.isnan(stacked), axis=0)
    cleaned_arrays = tuple(arr[mask] for arr in arrays)

    return cleaned_arrays


def log_scale_vs_linear_scale(
    x: np.ndarray,
    y: np.ndarray,
    x_range: typing.Tuple[float, float],
    y_range: typing.Tuple[float, float],
    x_log_scale: bool,
    y_log_scale: bool,
    n_x_bins: int,
    n_y_bins: int,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine the edge positions of the bins in x and y direction according to the
    choice of scale, i.e., log scale vs linear scale, for a given number of bins
    in both directions.

    Args:
        x (np.ndarray): x coordinates of the points.
        y (np.ndarray): y coordinates of the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density map.
        n_y_bins (int): Number of vertical bins for the density map.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple object containing the following arrays:
            - Coordinates x of the points (in log if x_log_scale is True);
            - Coordinates y of the points (in log if y_log_scale is True);
            - Edges of the bins in x direction according to the chosen scale.
            - Edges of the bins in y direction according to the chosen scale.
    """

    # Apply log scaling if requested.
    if x_log_scale:
        check_range_above_zero(x_range)
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_log_scale:
        check_range_above_zero(y_range)
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    return x, y, x_edges, y_edges
