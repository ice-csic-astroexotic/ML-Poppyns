"""
Performing a bilinear interpolation on a 2D array of data.

Authors:

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

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def bilinear_interpolation(
    x_in: np.ndarray,
    y_in: np.ndarray,
    f_in: np.ndarray,
    x_out: np.ndarray,
    y_out: np.ndarray,
) -> np.ndarray:
    """
    Function to perform bilinear interpolation on a 2D table of data.
    This function has been taken from the following website:
    https://pyquestions.com/how-to-perform-bilinear-interpolation-in-python

    Args:
        x_in (np.ndarray): array of x coordinates where the table is defined.
        y_in (np.ndarray): array of y coordinates where the table is defined.
        f_in (np.ndarray): 2D array of values defined on the x_in, y_in coordinates.
        x_out (np.ndarray): array of x coordinates where to compute the interpolated values.
        y_out (np.ndarray): array of y coordinates where to compute the interpolated values.

    Returns:
        (np.ndarray): 2D array of values interpolated on the x_out, y_out coordinates.
    """
    # Initialize the 2D array where to save the interpolated values.
    f_out = np.zeros((y_out.size, x_out.size), dtype=np.float64)

    for i in prange(f_out.shape[1]):
        # Find the coordinates x1 and x2 around the value x_out[i].
        idx = np.searchsorted(x_in, x_out[i])

        x1 = x_in[idx - 1]
        x2 = x_in[idx]
        x = x_out[i]

        for j in prange(f_out.shape[0]):
            # Find the coordinates y1 and y2 around the value y_out[i].
            idy = np.searchsorted(y_in, y_out[j])
            y1 = y_in[idy - 1]
            y2 = y_in[idy]
            y = y_out[j]

            # Take the values of the 2D array at the coordinate points:
            # (x1, y1), (x2, y1), (x1, y2), (x2, y2)
            # surrounding the point (x_out[i], y_out[i]) that we want to interpolate.
            f11 = f_in[idy - 1, idx - 1]
            f21 = f_in[idy - 1, idx]
            f12 = f_in[idy, idx - 1]
            f22 = f_in[idy, idx]

            # Interpolate with a bilinear method.
            # See also section "Repeated linear interpolation" in
            # https://en.wikipedia.org/wiki/Bilinear_interpolation.
            f_out[j, i] = (
                f11 * (x2 - x) * (y2 - y)
                + f21 * (x - x1) * (y2 - y)
                + f12 * (x2 - x) * (y - y1)
                + f22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))

    return f_out
