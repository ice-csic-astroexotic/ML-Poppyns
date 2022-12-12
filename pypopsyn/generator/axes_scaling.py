"""
Adjusting bin edges for a specific scaling of the axes.

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

import numpy as np


def check_range_above_zero(r_range: typing.Tuple[float, float]) -> None:
    """
    Check that a range larger than zero is provided.

    Args:
        r_range (float, float): value range to be checked.

    Returns:
        Returns None if r_range[0] and r_range[1] greater than 0,
        otherwise raises ValueError.
    """
    if (r_range[0] <= 0) or (r_range[1] <= 0):
        raise ValueError("Value range has to be above zero.")


def log_scale_vs_linear_scale(
    x_range: typing.Tuple[float, float],
    y_range: typing.Tuple[float, float],
    x_log_scale: bool,
    y_log_scale: bool,
    n_x_bins: int,
    n_y_bins: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Determine the edge positions of the bins in x and y direction according to the
    choice of scale, i.e., log scale vs linear scale, for a given number of bins
    in both directions.

    Args:
        x_range (float, float): horizontal range of values for the points.
        y_range (float, float): vertical range of values for the points.
        x_log_scale (bool): if True set the x-axis scale to log scale.
        y_log_scale (bool): if True set the y-axis scale to log scale.
        n_x_bins (int): number of horizontal bins for the density map.
        n_y_bins (int): number of vertical bins for the density map.

    Returns:
        (float, float): edges of the bins in x and y direction according to
        the chosen scale.

    """

    if x_log_scale and y_log_scale:
        check_range_above_zero(x_range)
        check_range_above_zero(y_range)
        x_edges = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        y_edges = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )

    elif x_log_scale and (y_log_scale is False):
        check_range_above_zero(x_range)
        x_edges = np.logspace(
            np.log10(x_range[0]), np.log10(x_range[1]), n_x_bins + 1
        )
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    elif (x_log_scale is False) and y_log_scale:
        check_range_above_zero(y_range)
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        y_edges = np.logspace(
            np.log10(y_range[0]), np.log10(y_range[1]), n_y_bins + 1
        )

    else:
        x_edges = np.linspace(x_range[0], x_range[1], n_x_bins + 1)
        y_edges = np.linspace(y_range[0], y_range[1], n_y_bins + 1)

    return x_edges, y_edges
