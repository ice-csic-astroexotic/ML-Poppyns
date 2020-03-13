""" Dataset generator functions

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters


def merge_dict(dict1: dict, dict2: dict) -> dict:
    """
    Merge dictionaries and keep values of common keys in list

    Args:
        dict1 (dict): first dictionary
        dict2 (dict):  second dictionary

    Returns:
        (dict): dictionary resulting from the merging of dict1 and dict2

    """
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [dict1[key], value]

    return dict3


def generate_density_map(
    x: np.array,
    x_range: typing.Tuple[float, float],
    y: np.array,
    y_range: typing.Tuple[float, float],
    filename: str,
    log_scale: bool = True,
    n_bins: int = 128,
    figure_width: int = 512,
    figure_height: int = 512,
) -> None:
    """
    Density map generator.

    Creates a density or heat map of a distribution of points given their
    X/Y coordinates in a 2D space. The heatmap is generated in log10 scale for
    both axes. The resulting image is written to disk.

    Args:
        x: horizontal coordinate values for the points.
        x_range: horizontal range of values for the points.
        y: vertical coordinate values for the points.
        y_range: vertical range of values for the points.
        filename: file path to generate the heatmap image.
        n_bins: number of vertical and horizontal bins for the heatmap.
        figure_width: width in pixels for the output figure.
        figure_height: height in pixels for the output figure.

    Returns:
        Nothing. An image is generated in the specified file path.

    """

    x_bins = None
    y_bins = None

    if log_scale:

        x_bins = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_bins
        )
        y_bins = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_bins
        )

    else:

        x_bins = np.linspace(x_range[0], x_range[1], n_bins)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins)

    density, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    density = scipy.ndimage.filters.gaussian_filter(density, sigma=1)

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(figure_width / DPI, figure_height / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.pcolormesh(x_edges, y_edges, density.T)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if log_scale:

        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.savefig(filename, dpi=DPI)
    plt.close(fig)
