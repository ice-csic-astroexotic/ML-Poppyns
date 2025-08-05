"""
    Calculating the cumulative distribution function for a given probability density
    function using the trapezoidal rule and drawing random values from the cumulative
    distribution and probability density function.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

from typing import Callable, Tuple

import numpy as np
import scipy.integrate as integrate


def cdf_calculator(
    x: np.ndarray, pdf: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Calculating the cumulative distribution function for any given probability density
    function evaluated at the points x using the trapezoidal rule.

    Args:
        x (np.ndarray): Discrete set of values at which the pdf is evaluated.
        pdf (Callable): Probability density function.

    Returns:
        (np.ndarray): Normalized cumulative distribution function.
    """

    cdf = integrate.cumulative_trapezoid(pdf(x), x, initial=0)
    cdf = cdf / np.max(cdf)

    return cdf


def random_from_cdf(
    x: np.ndarray, cdf: np.ndarray, num_draw: int
) -> np.ndarray:
    """
    Drawing random values from a given normalized cumulative distribution function.

    Args:
        x (np.ndarray): Discrete set of values at which the cdf is evaluated.
        cdf (np.ndarray): Normalized cumulative probability density function.
        num_draw (int): Number of values to draw.

    Returns:
        (np.ndarray): Random values drawn from the cdf.
    """

    cdf_rand = np.random.uniform(0, 1, num_draw)
    x_rand = np.interp(cdf_rand, cdf, x)

    return x_rand


def random_from_pdf(
    x: np.ndarray,
    pdf: Callable[[np.ndarray], np.ndarray],
    num_draw: int,
) -> np.ndarray:
    """
    Drawing random values from a given probability density function.

    Args:
        x (np.ndarray): Discrete set of values at which the pdf is evaluated.
        pdf (Callable): Probability density function.
        num_draw (int): Number of values to draw.

    Returns:
        (np.ndarray): Random values drawn from the pdf.
    """

    cdf = cdf_calculator(x, pdf)
    x_rand = random_from_cdf(x, cdf, num_draw)

    return x_rand


def random_from_pdf_2d(
    x1: np.ndarray,
    x2: np.ndarray,
    pdf_2d: np.ndarray,
    num_draw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drawing random values from a given 2D probability density function.
    x1 is the variable running along the rows (axis=0), x2 is the variable running along
    the columns (axis=1) of the 2D array defining the pdf.

    Args:
        x1 (np.ndarray): Discrete set of values for coordinate x1 at which the pdf is evaluated.
        x2 (np.ndarray): Discrete set of values for coordinate x2 at which the pdf is evaluated.
        pdf_2d (np.ndarray): 2D probability density function.
        num_draw (int): Number of values to draw.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Random points of coordinates (x1, x2) drawn from the pdf.
    """

    # Build the cumulative function grid by computing a cumulative function
    # for each column (i.e., for each value of x2) over axis=0.
    cum_func_grid = integrate.cumulative_trapezoid(
        pdf_2d, x1, axis=0, initial=0
    )

    # Compute the cumulative density function of the last row of the cumulative function grid
    # to find the total cdf for variable x2.
    cdf_x2 = integrate.cumulative_trapezoid(
        cum_func_grid[-1, :], x2, initial=0
    )
    cdf_x2 = cdf_x2 / np.max(cdf_x2)

    # Normalize the cumulative function grid, column-wise, to find the cdf for x1 given a value of x2.
    cdf_x1x2 = cum_func_grid / cum_func_grid.max(axis=0)

    # Draw a random x2 value.
    x2_rand = random_from_cdf(x2, cdf_x2, num_draw)

    # Find the indices of the cdfs for x1 corresponding to the values of x2 just drawn.
    idx = np.floor(
        (x2_rand - np.min(x2)) / (np.max(x2) - np.min(x2)) * len(x2)
    )
    idx = np.array(idx, dtype=int)

    # Draw a random x1 from the cdfs corresponding to the given x2 values.
    x1_rand = np.zeros_like(x2_rand)
    for i in range(len(x2_rand)):
        x1_rand[i] = random_from_cdf(x1, cdf_x1x2[:, idx[i]], 1)

    return x1_rand, x2_rand
