"""
    Dataset generator functions for 2D maps.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
        Vanessa Graber (graber@ice.csic.es)
        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import pypopsyn.generator.maps.axes_scaling as axs


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
    X/Y coordinates in a 2D space. The resulting image is generated in
    the specified file path.

    Args:
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        filename (str): File path to generate the density map image.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density map.
        n_y_bins (int): Number of vertical bins for the density map.
        colormap (str): Colormap to use for the image.
    """

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
        ax.set_xlim(np.log10(x_range[0]), np.log10(x_range[1]))
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        ax.set_xlim(x_range[0], x_range[1])

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
        ax.set_ylim(np.log10(y_range[0]), np.log10(y_range[1]))
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)
        ax.set_ylim(y_range[0], y_range[1])

    # Generating a 2D histogram that counts the number of objects contained
    # in each respective pixel; following the discrete count, we apply a
    # Gaussian filter to smear out the hard edges of the distribution to
    # improve the stability of the machine learning framework;
    # x (y) values are histogrammed along first (second) dimension.
    density, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    density = scipy.ndimage.gaussian_filter(density, sigma=1)

    # Generating a pseudocolor plot of the smeared out density distribution;
    # we transpose the array as pcolormesh is indexed starting from the lower
    # left , i.e., the column (row) index corresponds to the x (y) coordinate.
    ax.pcolormesh(x_edges, y_edges, density.T, cmap=colormap)

    fig.savefig(filename, dpi=DPI)
    plt.close(fig)


def generate_avg_weight_map(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    w: np.ndarray,
    w_avg_min: float,
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
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        w (np.ndarray): Weight values for the points.
        w_avg_min (float): Minimum average weight value for the bins.
        filename (str): File path to generate the heat map image.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the weight map.
        n_y_bins (int): Number of vertical bins for the weight map.
        colormap (str): Colormap to use for the image.
    """

    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
        ax.set_xlim(np.log10(x_range[0]), np.log10(x_range[1]))
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        ax.set_xlim(x_range[0], x_range[1])

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
        ax.set_ylim(np.log10(y_range[0]), np.log10(y_range[1]))
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)
        ax.set_ylim(y_range[0], y_range[1])

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
    # the final result as the original array elements are zero anyway.
    total_per_bin[total_per_bin == 0] = 0.0001
    avg_weight = total_weight / total_per_bin

    # If there are no pulsars in a given area, we will set the value in the bin to w_avg_min.
    # w_avg_min should be set to a value that guarantees a good background value for the areas that have no pulsars.
    avg_weight[total_per_bin == 0.0001] = w_avg_min

    # To avoid potential sharp edges, we apply a Gaussian filter.
    avg_weight = scipy.ndimage.gaussian_filter(avg_weight, sigma=1)

    # Generating a pseudocolor plot of the smeared out average distribution;
    # we transpose the array as pcolormesh is indexed starting from the lower
    # left , i.e., the column (row) index corresponds to the x (y) coordinate.
    ax.pcolormesh(x_edges, y_edges, avg_weight.T, cmap=colormap)

    fig.savefig(filename, dpi=DPI)

    plt.close(fig)


def generate_kde_density_map(
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
    KDE density map generator.

    Creates a density or heat map of a distribution of points, computing the kernel-density estimation (KDE) given the
    point X/Y coordinates in a 2D space. The resulting image is written to disk.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (Tuple[float, float]): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (Tuple[float, float]): vertical range of values for the points.
        filename (str): file path to generate the density map image.
        x_log_scale (bool): if True set the x axis scale to log scale.
        y_log_scale (bool): if True set the y axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density map.
        n_y_bins (int): number of vertical bins for the density map.
        colormap (str): colormap to use for the image.
    """
    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
        ax.set_xlim(np.log10(x_range[0]), np.log10(x_range[1]))
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        ax.set_xlim(x_range[0], x_range[1])

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
        ax.set_ylim(np.log10(y_range[0]), np.log10(y_range[1]))
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)
        ax.set_ylim(y_range[0], y_range[1])

    # Create a meshgrid where to compute the KDE.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Construct the KDE function.
    points = np.vstack([x, y])
    kde = scipy.stats.gaussian_kde(points)

    # Evaluate the KDE in each position points and restructure the flat output of KDE back into a 2D grid
    # that corresponds to the x-y coordinate grid.
    density = kde(positions).reshape(len(y_centers), len(x_centers))

    # Generating a pseudocolor plot of the density distribution.
    ax.pcolormesh(x_edges, y_edges, density, cmap=colormap)

    fig.savefig(filename, dpi=DPI)

    plt.close(fig)


def generate_kde_weight_map(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    w: np.ndarray,
    w_min: float,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 128,
    n_y_bins: int = 128,
    colormap: str = "Greys",
) -> None:
    """
    Weighted KDE density map generator.

    Creates a weighted density or heat map of a distribution of points, computing the weighted kernel-density estimation
    (KDE) given the point X/Y coordinates in a 2D space and some weights. The resulting image is written to disk.

    Args:
        x (np.ndarray): horizontal coordinate values for the points.
        x_range (Tuple[float, float]): horizontal range of values for the points.
        y (np.ndarray): vertical coordinate values for the points.
        y_range (Tuple[float, float]): vertical range of values for the points.
        w (np.ndarray): Weight values for the points.
        w_min (float): Minimum weight to subtract to ensure positivity.
        filename (str): file path to generate the density map image.
        x_log_scale (bool): if True set the x axis scale to log scale.
        y_log_scale (bool): if True set the y axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density map.
        n_y_bins (int): number of vertical bins for the density map.
        colormap (str): colormap to use for the image.
    """
    DPI = 512
    fig = plt.figure(dpi=DPI, frameon=False)
    fig.set_size_inches(n_x_bins / DPI, n_y_bins / DPI)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Apply log scaling if requested
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
        ax.set_xlim(np.log10(x_range[0]), np.log10(x_range[1]))
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        ax.set_xlim(x_range[0], x_range[1])

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
        ax.set_ylim(np.log10(y_range[0]), np.log10(y_range[1]))
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)
        ax.set_ylim(y_range[0], y_range[1])

    # Rescale weights to be strictly positive
    positive_w = w - w_min

    # Create a meshgrid where to compute the KDE.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Construct the weighted KDE function.
    points = np.vstack([x, y])
    kde = scipy.stats.gaussian_kde(points, weights=positive_w)

    # Evaluate the KDE in each position points and restructure the flat output of KDE back into a 2D grid
    # that corresponds to the x-y coordinate grid.
    weighted_density = kde(positions).reshape(n_y_bins, n_x_bins)

    # Generating a pseudocolor plot of the weighted density distribution.
    ax.pcolormesh(x_edges, y_edges, weighted_density, cmap=colormap)

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
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        filename (str): File path to generate the density matrix.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density matrix.
        n_y_bins (int): Number of vertical bins for the density matrix.
    """

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

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
    w_avg_min: float,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 32,
    n_y_bins: int = 32,
) -> None:
    """
    Average weighted matrix generator.

    Creates a matrix of the average weight w of a distribution of points given their
    X/Y coordinates in a 2D space.
    The resulting matrix shows the average value of the weights in each bin.

    Args:
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        w (np.ndarray): Weight values for the points.
        w_avg_min (float): Minimum average weight value for the bins.
        filename (str): File path to generate the density matrix.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density matrix.
        n_y_bins (int): Number of vertical bins for the density matrix.
    """

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

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
    # the final result as the original array elements are zero anyway.
    total_per_bin[total_per_bin == 0] = 0.0001
    avg_weight = total_weight / total_per_bin

    # If there are no pulsars in a given area, we will set the value in the bin to w_avg_min.
    # w_avg_min should be set to a value that guarantees a good background value for the areas that have no pulsars.
    avg_weight[total_per_bin == 0.0001] = w_avg_min

    # To avoid potential sharp edges, we apply a Gaussian filter.
    avg_weight = scipy.ndimage.gaussian_filter(avg_weight, sigma=1)

    np.save(filename, avg_weight)


def generate_kde_density_matrix(
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
    KDE density matrix generator.

    Creates a density matrix of distribution of points, computing the kernel-density estimation (KDE) given the
    point X/Y coordinates in a 2D space. The resulting matrix is saved as .npy file.

    Args:
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        filename (str): File path to generate the density matrix.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density matrix.
        n_y_bins (int): Number of vertical bins for the density matrix.
    """

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    # Create a meshgrid where to compute the KDE.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Construct the KDE function.
    points = np.vstack([x, y])
    kde = scipy.stats.gaussian_kde(points)

    # Evaluate the KDE in each position points and restructure the flat output of KDE back into a 2D grid
    # that corresponds to the x-y coordinate grid.
    density = kde(positions).reshape(len(y_centers), len(x_centers))

    np.save(filename, density)


def generate_kde_weight_matrix(
    x: np.ndarray,
    x_range: typing.Tuple[float, float],
    y: np.ndarray,
    y_range: typing.Tuple[float, float],
    w: np.ndarray,
    w_min: float,
    filename: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    n_x_bins: int = 32,
    n_y_bins: int = 32,
) -> None:
    """
    Weighted KDE density map generator.

    Creates a weighted density matrix of a distribution of points, computing the weighted kernel-density estimation
    (KDE) given the point X/Y coordinates in a 2D space and some weights. The resulting matrix is saved as .npy file.

    Args:
        x (np.ndarray): Horizontal coordinate values for the points.
        x_range (Tuple[float, float]): Horizontal range of values for the points.
        y (np.ndarray): Vertical coordinate values for the points.
        y_range (Tuple[float, float]): Vertical range of values for the points.
        w (np.ndarray): Weight values for the points.
        w_min (float): Minimum weight to subtract to ensure positivity.
        filename (str): File path to generate the density matrix.
        x_log_scale (bool): If True set the x-axis scale to log scale.
        y_log_scale (bool): If True set the y-axis scale to log scale.
        n_x_bins (int): Number of horizontal bins for the density matrix.
        n_y_bins (int): Number of vertical bins for the density matrix.
    """

    # Apply log scaling if requested.
    if x_log_scale:
        x_edges = np.linspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        x = np.log10(x)
    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)

    if y_log_scale:
        y_edges = np.linspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )
        y = np.log10(y)
    else:
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    # Rescale weights to be strictly positive
    positive_w = w - w_min

    # Create a meshgrid where to compute the KDE.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Construct the weighted KDE function.
    points = np.vstack([x, y])
    kde = scipy.stats.gaussian_kde(points, weights=positive_w)

    # Evaluate the KDE in each position points and restructure the flat output of KDE back into a 2D grid
    # that corresponds to the x-y coordinate grid.
    weighted_density = kde(positions).reshape(n_y_bins, n_x_bins)

    np.save(filename, weighted_density)
