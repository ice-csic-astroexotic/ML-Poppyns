"""
Initial velocity distribution for the stellar population.

The velocity is composed of two contributions, the neutron stars' proper motion
caused by kicks during the supernova as well as the motion of the galaxy itself.
For the former, we follow Gullon et al. (2014).
"""

import numpy as np


def pdf_proper_velocity(v: float) -> float:
    """
    Probability distribution function for the neutron stars' proper velocities.

    Args:
        v (float): proper velocity in km/s

    Returns:
        float: stellar proper velocity distribution in 1/(km/s)
    """

    # we follow Gullon et al. (2014) and consider an exponential distribution

    v_mean = 600.0  # [km/s]
    p_v = 1.0 / v_mean * np.exp(-v / v_mean)

    return p_v
