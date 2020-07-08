import argparse
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

start = time.time()

###########################################################################################


def get_intersections(
    x0: float, y0: float, r0: float, x1: float, y1: float, r1: float
) -> list:
    """
    Gives the intersection points between two circles.
    Args:
        x0 (float): x coordinate of the center of the first circle.
        y0 (float): y coordinate of the center of the first circle.
        r0 (float): radius of the first circle.
        x1 (float): x coordinate of the center of the second circle.
        y1 (float): y coordinate of the center of the second circle.
        r1 (float): radius of the second circle.
    Returns:
        (float, float, float, float): x and y coordinates of the two intersection points between the two circles.
    """

    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # The two circles are non intersecting.
    if d > r0 + r1:
        return None
        # One circle is inside the other.
    elif d - abs(r0 - r1) < -1.0e-10:
        return None
        # The two circles are coincident.
    elif d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        if r0 ** 2 - a ** 2 < 1.0e-10:
            h = 0.0
        else:
            h = np.sqrt(r0 ** 2 - a ** 2)

        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d

        # Compute the coordinates of the two intersection points.
        # If the circles are tangent the two intersection points are coincident.

        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return [x3, y3, x4, y4]


def cell_radius(r: np.array, dr: float, res_r: int) -> np.array:
    """
    evaluate the radius of the circular cells. The radius increases linearly with the distance from the Sun
    Args:
        r (np.array): grid of radii of the heliocentric circles.
        dr (float): radial step of the r array.
        res_r (int): resolution used to define the lenght step dr = R_sun / res_r of the r array.
    Returns:
        (np.array): grid of radii of the circular cells.
    """

    L = len(r)
    i = np.arange(1, L + 1)

    # The following formula is a generalization of the formula in section 2 in Yusifov & Kucuck 2004.
    r_c = 0.5 * dr * ((np.max(r) / (res_r * dr) - 1.0) * (i - 1.0) / L + 1)

    return r_c


def construct_grid(
    res_R: int, res_r: int, R_sun: float
) -> Tuple[np.array, np.array, np.array]:
    """
    Construct a cell grid following the indication in section 2 in Yusifov & Kucuck 2004. r indicates the heliocentric distance, while R indicate the galactocentric distance.
    The final constructed grid is also plotted for visual check.
    Args:
        res_R (int): resolution used to define the radial step dR = R_sun / res_R of the R grid.
        res_r (int): resolution used to define the radial step dr = R_sun / res_r of the r grid.
        R_sun (float): radial distance of the Sun in kpc from the Galctic center.
    Returns:
        (np.array, np.array, np.array): arrays of coordinates x_c and y_c of the points of the grids
            and array of radii of the circular cells around each grid point.
    """
    # define the Galactic center coordinates and the Sun coordinates
    x_gc = 0.0
    y_gc = 0.0
    x_sun = 0.0
    y_sun = R_sun

    # define the radial steps of the grid and build the r and R grid.
    dR = R_sun / res_R
    dr = R_sun / res_r
    r_grid = np.arange(0.0, 20.0, dr)
    R_grid = np.arange(0.0, 20.0, dR)

    # build the grid for the cells radii.
    rc_grid = cell_radius(r_grid, dr, res_r)

    # initialize the output.
    x_grid = []
    y_grid = []
    r_c = []

    # initialize the plot.
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")

    # build the cell grid.
    for i in range(0, len(R_grid)):

        circle_R = plt.Circle(
            (x_gc, y_gc), R_grid[i], color="b", fill=False, rasterized=True
        )
        ax.add_artist(circle_R)

        for j in range(0, len(r_grid)):

            circle_r = plt.Circle(
                (x_sun, y_sun),
                r_grid[j],
                color="b",
                fill=False,
                rasterized=True,
            )
            ax.add_artist(circle_r)

            # find the intersection points between the galactocentric circles and the heliocentric circles.
            intersections = get_intersections(
                x_gc, y_gc, R_grid[i], x_sun, y_sun, r_grid[j]
            )

            if intersections is not None:
                # if the circles are tangent save only one intersection
                if (intersections[0] == intersections[2]) & (
                    intersections[1] == intersections[3]
                ):
                    x_grid.append(intersections[0])
                    y_grid.append(intersections[1])
                    r_c.append(rc_grid[j])
                    circle_c = plt.Circle(
                        (intersections[0], intersections[1]),
                        rc_grid[j],
                        color="black",
                        alpha=0.5,
                        fill=False,
                        rasterized=True,
                    )
                    ax.add_artist(circle_c)
                else:
                    x_grid.append(intersections[0])
                    x_grid.append(intersections[2])
                    y_grid.append(intersections[1])
                    y_grid.append(intersections[3])
                    r_c.append(rc_grid[j])
                    r_c.append(rc_grid[j])
                    circle_c1 = plt.Circle(
                        (intersections[0], intersections[1]),
                        rc_grid[j],
                        color="black",
                        alpha=0.5,
                        fill=False,
                        rasterized=True,
                    )
                    circle_c2 = plt.Circle(
                        (intersections[2], intersections[3]),
                        rc_grid[j],
                        color="black",
                        alpha=0.5,
                        fill=False,
                        rasterized=True,
                    )
                    ax.add_artist(circle_c1)
                    ax.add_artist(circle_c2)

    ax.plot(
        x_grid,
        y_grid,
        linestyle="None",
        marker="o",
        color="blue",
        markersize=5,
        alpha=0.5,
        rasterized=True,
    )
    ax.plot(
        0.0,
        R_sun,
        linestyle="None",
        marker="o",
        color="gold",
        markersize=10,
        alpha=0.5,
        rasterized=True,
    )

    plt.show(block=True)

    # export grid to a file
    df = pd.DataFrame({"x_c": x_grid, "y_c": y_grid, "r_c": r_c})

    df.columns = pd.MultiIndex.from_tuples(
        zip(df.columns, ["[kpc]", "[kpc]", "[kpc]"])
    )

    df.to_csv("grid.csv", index=False, header=True)

    return x_grid, y_grid, r_c


def main(args):
    construct_grid(
        args.res_R, args.res_r, args.R_sun,
    )


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Cell grid constructor")

    args.add_argument(
        "--res_R",
        type=int,
        default=5,
        help="resolution used to define the radial step dR = R_sun / res_R of the R grid.",
    )
    args.add_argument(
        "--res_r",
        type=int,
        default=5,
        help="resolution used to define the radial step dr = R_sun / res_r of the r grid.",
    )
    args.add_argument(
        "--R_sun",
        type=float,
        default=8.5,
        help="radial distance of the Sun in kpc from the Galctic center.",
    )
    args = args.parse_args()

    main(args)


###########################################################################################

end = time.time()
print("elapsed execution time: ", (end - start), "seconds")
