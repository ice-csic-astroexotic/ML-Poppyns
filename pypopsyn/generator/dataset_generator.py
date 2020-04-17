""" Dataset generator functions

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters


def generate_density_map(
    x: np.array,
    x_range: typing.Tuple[float, float],
    y: np.array,
    y_range: typing.Tuple[float, float],
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_bins: int = 128,
    figure_width: int = 512,
    figure_height: int = 512,
) -> None:
    """
    Density map generator.

    Creates a density or heat map of a distribution of points given their
    X/Y coordinates in a 2D space. The resulting image is written to disk.

    Args:
        x: horizontal coordinate values for the points.
        x_range: horizontal range of values for the points.
        y: vertical coordinate values for the points.
        y_range: vertical range of values for the points.
        filename: file path to generate the heatmap image.
        x_log_scale: if True set the x axis scale to log scale
        y_log_scale: if True set the y axis scale to log scale
        n_bins: number of vertical and horizontal bins for the heatmap.
        figure_width: width in pixels for the output figure.
        figure_height: height in pixels for the output figure.

    Returns:
        Nothing. An image is generated in the specified file path.

    """

    x_bins = None
    y_bins = None

    if x_log_scale and y_log_scale:
        x_bins = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_bins
        )
        y_bins = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_bins
        )

    elif x_log_scale and (y_log_scale is False):
        x_bins = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_bins
        )
        y_bins = np.linspace(y_range[0], y_range[1], n_bins)

    elif (x_log_scale is False) and y_log_scale:
        x_bins = np.linspace(x_range[0], x_range[1], n_bins)
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

    ax.pcolormesh(x_edges, y_edges, density.T, cmap="Greys")
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if x_log_scale:
        ax.set_xscale("log")

    if y_log_scale:
        ax.set_yscale("log")

    fig.savefig(filename, dpi=DPI)
    plt.close(fig)


def generate_avg_weight_histo2D(
    x: np.array,
    x_range: typing.Tuple[float, float],
    y: np.array,
    y_range: typing.Tuple[float, float],
    w: np.array,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_bins: int = 128,
    figure_width: int = 512,
    figure_height: int = 512,
) -> None:
    """
    multichannel weighted 2D histogram generator.

    Creates a 2D histogram of a distribution of points given their
    X/Y coordinates in a 2D space with weights w.
    The resulting histogram for each bin shows the average value of the weights in
    each bin.

    Args:
        x: horizontal coordinate values for the points.
        x_range: horizontal range of values for the points.
        y: vertical coordinate values for the points.
        y_range: vertical range of values for the points.
        w: weight values for the points.
        filename: file path to generate the heatmap image.
        x_log_scale: if True set the x axis scale to log scale
        y_log_scale: if True set the y axis scale to log scale
        n_bins: number of vertical and horizontal bins for the heatmap.
        figure_width: width in pixels for the output figure.
        figure_height: height in pixels for the output figure.

    Returns:
        Nothing. A 2D histogram is generated.

    """

    x_bins = None
    y_bins = None

    if x_log_scale and y_log_scale:
        x_bins = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_bins
        )
        y_bins = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_bins
        )

    elif x_log_scale and (y_log_scale is False):
        x_bins = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_bins
        )
        y_bins = np.linspace(y_range[0], y_range[1], n_bins)

    elif (x_log_scale is False) and y_log_scale:
        x_bins = np.linspace(x_range[0], x_range[1], n_bins)
        y_bins = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_bins
        )

    else:
        x_bins = np.linspace(x_range[0], x_range[1], n_bins)
        y_bins = np.linspace(y_range[0], y_range[1], n_bins)

    # if weights are the velocity component that can be also negative we need to take
    # the absolute value
    total_per_bin, x_edges, y_edges = np.histogram2d(
        x, y, bins=[x_bins, y_bins]
    )
    total_weight, x_edges, y_edges = np.histogram2d(
        x, y, bins=[x_bins, y_bins], weights=w
    )

    # change 0 to 0.0001 in order to avoid dividing by 0
    total_per_bin[total_per_bin == 0] = 0.0001

    avg_weight = total_weight / total_per_bin

    avg_weight = scipy.ndimage.filters.gaussian_filter(avg_weight, sigma=1)

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(figure_width / DPI, figure_height / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.pcolormesh(x_edges, y_edges, avg_weight.T, cmap="Greys")
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if x_log_scale:
        ax.set_xscale("log")

    if y_log_scale:
        ax.set_yscale("log")

    fig.savefig(filename, dpi=DPI)

    plt.close(fig)
