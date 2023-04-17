"""
Dataset generator functions.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)

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

import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters

import pypopsyn.generator.axes_scaling as axs


def generate_density_map(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 128,
    n_y_bins: int = 128,
    colormap: str = "Greys",
) -> None:
    """
    Density map generator.

    Creates a density or heat map of a distribution of points given their
    X/Y coordinates in a 2D space. The resulting image is written to disk.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (float, float): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (float, float): vertical range of values for the points.
        filename (str): file path to generate the density map image.
        x_log_scale (bool): if True set the x-axis scale to log scale.
        y_log_scale (bool): if True set the y-axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density map.
        n_y_bins (int): number of vertical bins for the density map.
        colormap (str): colormap to use for the image.

    Returns:
        Nothing. An image is generated in the specified file path.

    """

    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        x_range,
        y_range,
        x_log_scale,
        y_log_scale,
        n_x_bins,
        n_y_bins,
    )

    # Generating a 2D histogram that counts the number of objects contained
    # in each respective pixel; following the discrete count, we apply a
    # Gaussian filter to smear out the hard edges of the distribution to
    # improve the stability of the machine learning framework;
    # x (y) values are histogrammed along first (second) dimension.
    density, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    density = scipy.ndimage.gaussian_filter(density, sigma=1)

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Generating a pseudocolor plot of the smeared out density distribution;
    # we transpose the array as pcolormesh is indexed starting from the lower
    # left , i.e., the column (row) index corresponds to the x (y) coordinate.
    ax.pcolormesh(x_edges, y_edges, density.T, cmap=colormap)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if x_log_scale:
        ax.set_xscale("log")

    if y_log_scale:
        ax.set_yscale("log")

    fig.savefig(filename, dpi=DPI)
    plt.close(fig)


def generate_avg_weight_map(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    w: np.ndarray,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 128,
    n_y_bins: int = 128,
    colormap: str = "Greys",
) -> None:
    """
    Average weighted map generator.

    Creates a map of the average weight w of a distribution of points given their
    X/Y coordinates in a 2D space.
    The resulting map shows the average value of the weights in each bin.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (float, float): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (float, float): vertical range of values for the points.
        w (np.ndarray): weight values for the points.
        filename (str): file path to generate the heat map image.
        x_log_scale (bool): if True set the x-axis scale to log scale.
        y_log_scale (bool): if True set the y-axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the weight map.
        n_y_bins (int): number of vertical bins for the weight map.
        colormap (str): colormap to use for the image.

    Returns:
        Nothing. An image is generated in the specified file path.

    """

    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        x_range,
        y_range,
        x_log_scale,
        y_log_scale,
        n_x_bins,
        n_y_bins,
    )

    # If the quantity desired as the weight can become negative, e.g.,
    # one of the velocity components, take the absolute value and use that
    # as the weight to avoid the possibility of summing to zero.
    total_per_bin, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    total_weight, x_edges, y_edges = np.histogram2d(
        x, y, bins=[x_edges, y_edges], weights=w
    )

    # Dividing the total summed weight per bin by the number of objects to
    # obtain the average value per pixel; to avoid dividing by 0, we change
    # the values in total_weight from 0 to 0.0001; doing so does not affect
    # the final result as the original array elements are zero anyway;
    # to avoid potential sharp edges, we apply a Gaussian filter.
    total_per_bin[total_per_bin == 0] = 0.0001
    avg_weight = total_weight / total_per_bin
    avg_weight = scipy.ndimage.gaussian_filter(avg_weight, sigma=1)

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Generating a pseudocolor plot of the smeared out average distribution;
    # we transpose the array as pcolormesh is indexed starting from the lower
    # left , i.e., the column (row) index corresponds to the x (y) coordinate.
    ax.pcolormesh(x_edges, y_edges, avg_weight.T, cmap=colormap)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if x_log_scale:
        ax.set_xscale("log")

    if y_log_scale:
        ax.set_yscale("log")

    fig.savefig(filename, dpi=DPI)

    plt.close(fig)


def generate_density_matrix(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 128,
    n_y_bins: int = 128,
) -> None:
    """
    Density matrix generator.

    Creates a density matrix of a distribution of points given their
    X/Y coordinates in a 2D space. The resulting matrix is saved as .npy file.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (float, float): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (float, float): vertical range of values for the points.
        filename (str): file path to generate the density matrix.
        x_log_scale (bool): if True set the x-axis scale to log scale.
        y_log_scale (bool): if True set the y-axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density matrix.
        n_y_bins (int): number of vertical bins for the density matrix.

    Returns:
        Nothing. A NumPy 2D array is generated in the specified file path.

    """

    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        x_range,
        y_range,
        x_log_scale,
        y_log_scale,
        n_x_bins,
        n_y_bins,
    )

    # Generating a 2D histogram that counts the number of objects contained
    # in each respective bin; x (y) values are histogrammed along first
    # (second) dimension; to avoid potential sharp edges, we apply a Gaussian filter.
    density, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    density = scipy.ndimage.gaussian_filter(density, sigma=1)

    np.save(filename, density)


def generate_avg_weight_matrix(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    w: np.ndarray,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 128,
    n_y_bins: int = 128,
) -> None:
    """
    Average weighted matrix generator.

    Creates a matrix of the average weight w of a distribution of points given their
    X/Y coordinates in a 2D space.
    The resulting matrix shows the average value of the weights in each bin.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (float, float): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (float, float): vertical range of values for the points.
        w (np.ndarray): weight values for the points.
        filename (str): file path to generate the density matrix.
        x_log_scale (bool): if True set the x-axis scale to log scale.
        y_log_scale (bool): if True set the y-axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density matrix.
        n_y_bins (int): number of vertical bins for the density matrix.

    Returns:
        Nothing. A NumPy 2D array is generated in the specified file path.

    """

    x_edges, y_edges = axs.log_scale_vs_linear_scale(
        x_range,
        y_range,
        x_log_scale,
        y_log_scale,
        n_x_bins,
        n_y_bins,
    )

    # If the quantity desired as the weight can become negative, e.g.,
    # one of the velocity components, take the absolute value and use that
    # as the weight to avoid the possibility of summing to zero.
    total_per_bin, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    total_weight, x_edges, y_edges = np.histogram2d(
        x, y, bins=[x_edges, y_edges], weights=w
    )

    # Dividing the total summed weight per bin by the number of objects to
    # obtain the average value per bin; to avoid dividing by 0, we change
    # the values in total_weight from 0 to 0.0001; doing so does not affect
    # the final result as the original array elements are zero anyway;
    # to avoid potential sharp edges, we apply a Gaussian filter.
    total_per_bin[total_per_bin == 0] = 0.0001
    avg_weight = total_weight / total_per_bin
    avg_weight = scipy.ndimage.gaussian_filter(avg_weight, sigma=1)

    np.save(filename, avg_weight)
