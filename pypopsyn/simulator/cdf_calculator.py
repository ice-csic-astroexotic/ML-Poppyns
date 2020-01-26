"""
Calculating the cumulative distribution function using the trapezoidal rule
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
