"""
Calculating the cumulative distribution function using the trapezoidal rule

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)
"""

from typing import Callable

import numpy as np
import scipy.integrate as integrate


def cdf_calculator(x: np.ndarray, pdf: Callable[[float], float]) -> np.ndarray:
    """
    Calculating the cumulative distribution function for any given probability density
    function evaluated at the points x using the trapezoidal rule.

    Args:
        x (np.ndarray): discrete set of values at which the pdf, cdf is evaluated
        pdf (Callable): probability distribution function

    Returns:
        np.ndarray: normalised cumulative distribution function
    """

    # vectorising the pdf to take in an array
    pdf_vect = np.vectorize(pdf)

    cdf = integrate.cumtrapz(pdf_vect(x), x, initial=0)
    cdf = cdf / np.max(cdf)

    return cdf


def random_from_cdf(
    x: np.ndarray, cdf: np.ndarray, num_draw: (int)
) -> np.ndarray:
    """
    Drawing a random number value from a given normalized cumulative distribution
    function corresponding to any given probability density function.

    Args:
        x (np.ndarray): discrete set of values at which the pdf, cdf is evaluated
        cdf (np.ndarray): normalized cumulative probability distribution function
        num_draw (int): number of values to draw

    Returns:
        np.ndarray: random values drawn from the pdf
    """

    cdf_rand = np.random.uniform(0, 1, num_draw)
    x_rand = np.interp(cdf_rand, cdf, x)

    return x_rand


def random_from_pdf(
    x: np.ndarray, pdf: Callable[[float], float], num_draw: (int)
) -> np.ndarray:
    """
    Drawing a random number value from a given probability density function.

    Args:
        x (np.ndarray): discrete set of values at which the pdf is evaluated
        pdf (Callable): probability distribution function
        num_draw (int): number of values to draw

    Returns:
        np.ndarray: random values drawn from the pdf
    """

    cdf = cdf_calculator(x, pdf)
    x_rand = random_from_cdf(x, cdf, num_draw)

    return x_rand
