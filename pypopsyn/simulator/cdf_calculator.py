"""
Calculating the cumulative distribution function for a given probability density
function using the trapezoidal rule and drawing random values from the cumulative
distribution and probability density function.

Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)

MIT License

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

from typing import Callable

import numpy as np
import scipy.integrate as integrate


def cdf_calculator(x: np.ndarray, pdf: Callable[[float], float]) -> np.ndarray:
    """
    Calculating the cumulative distribution function for any given probability density
    function evaluated at the points x using the trapezoidal rule.

    Args:
        x (np.ndarray): discrete set of values at which the pdf is evaluated.
        pdf (Callable): probability density function.

    Returns:
        np.ndarray: normalized cumulative distribution function.
    """

    # Vectorizing the pdf to take in an array.
    pdf_vect = np.vectorize(pdf)

    cdf = integrate.cumtrapz(pdf_vect(x), x, initial=0)
    cdf = cdf / np.max(cdf)

    return cdf


def random_from_cdf(
    x: np.ndarray, cdf: np.ndarray, num_draw: int, seed: int = None
) -> np.ndarray:
    """
    Drawing random values from a given normalized cumulative distribution function.

    Args:
        x (np.ndarray): discrete set of values at which the cdf is evaluated.
        cdf (np.ndarray): normalized cumulative probability density function.
        num_draw (int): number of values to draw.
        seed (int): seed for random number generation,
        set to None unless otherwise specified.

    Returns:
        np.ndarray: random values drawn from the cdf.
    """

    np.random.seed(seed)

    cdf_rand = np.random.uniform(0, 1, num_draw)
    x_rand = np.interp(cdf_rand, cdf, x)

    return x_rand


def random_from_pdf(
    x: np.ndarray,
    pdf: Callable[[float], float],
    num_draw: int,
    seed: int = None,
) -> np.ndarray:
    """
    Drawing random values from a given probability density function.

    Args:
        x (np.ndarray): discrete set of values at which the pdf is evaluated.
        pdf (Callable): probability density function.
        num_draw (int): number of values to draw.
        seed (int): seed for random number generation in the random_from_cdf function,
        set to None unless otherwise specified.

    Returns:
        np.ndarray: random values drawn from the pdf.
    """

    cdf = cdf_calculator(x, pdf)
    x_rand = random_from_cdf(x, cdf, num_draw, seed)

    return x_rand
