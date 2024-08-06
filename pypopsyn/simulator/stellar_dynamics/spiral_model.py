"""
    Model for the Milky Way spiral arms.

    We consider two different models:

    1) saFK06: A galactic spiral structure according to eq. (12) of Faucher-Giguère & Kaspi (2006)
    (see also Wainscoat et al. 1992). Their model consists of four arms plus a Local arm from
    Wainscoat et al. (1992).

    2) saYMW17: A galactic spiral structure according to eq. (12) of Faucher-Giguère & Kaspi (2006)
    but with parameters re-adapted from Yao et al. (2017). Their model consists of four arms
    plus a Local arm from Hou et al. (2014).

    Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""

import random

import numpy as np

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
from pypopsyn.simulator.config_simulator import cfg

spiral_model = None


def initialize_spiral_model() -> None:
    """
    Initializing the spiral model employed in the simulation. We have implemented two
    versions, i.e., the spiral structure of Faucher-Giguère & Kaspi (2006) (with
    the addition of the Local arm from Wainscoat et al. (1992)) and the spiral model from
    Yao et al. (2017). The spiral_model variable is made available on a global level.
    """
    global spiral_model

    if cfg["spiral_arms"] == "saFK06":
        spiral_model = SpiralModelFK06()
    elif cfg["spiral_arms"] == "saYMW17":
        spiral_model = SpiralModelYMW17()
    else:
        raise ValueError(
            "The spiral arm model does not exist. Choose between saFK06 or saYMW17."
        )


class SpiralModelBase:
    """
    Base class for any spiral model to ensure that a common interface between all of
    them is respected. The following abstract methods must be implemented or an error
    will be raised.
    """

    def __init__(self):
        self.arm_param = dict
        self.arm_probability = []
        self.local_r_min = float
        self.local_r_max = float
        super().__init__()

    def generate_arm_index(
        self, arm_number: int, NS_number: int
    ) -> np.ndarray:
        """
        Generate an array of indices related to the spiral arm associated to each star
        according to a probability evaluated taking into account the radial distribution of stars
        and the limited extension of the Local arm with respect to the other arms.

        Args:
            arm_number (int): number of arms to simulate.
            NS_number (int): number of stars to simulate.

        Returns:
            (np.ndarray): array of random indices for the spiral arm associated to each star.

        """

        # Initialize the array of arm indices.
        arm_index_rand = np.zeros(NS_number)

        # If the Local arm is excluded, uniformly distribute the stars on the other arms.
        if arm_number == 4:
            arm_index_rand = np.random.randint(1, arm_number + 1, NS_number)

        # If the Local arm is included distribute the stars on the arms according to the probability
        # specified in arm_probability.
        elif arm_number == 5:
            choices = [1, 2, 3, 4, 5]
            arm_index_rand = np.array(
                random.choices(
                    choices, weights=self.arm_probability, k=NS_number
                )
            )

        return arm_index_rand

    def check_arm_index(self, arm_index: np.ndarray) -> None:
        """
        Check that the index for the spiral galaxy arms is not <1 or >5.

        Args:
            arm_index (np.ndarray): index for the respective spiral arms.
        """
        if np.any(arm_index < 1) or np.any(arm_index > 5):
            raise ValueError("One of arm indices is out of range.")

    def calculate_phi(
        self, r: np.ndarray, arm_index: np.ndarray
    ) -> np.ndarray:
        """
        Calculating the angular coordinates of a neutron star for a given distance
        from the galactic center incorporating the Milky Way's arm structure according
        to eq. (12) of Faucher-Giguère & Kaspi (2006) (see also Wainscoat et al. 1992).

        Args:
            r (np.ndarray): distances from the galactic center in [kpc].
            arm_index (np.ndarray): indices for the respective spiral arms,
                0 < arm_index < 6.

        Returns:
            (np.ndarray): galactocentric phi coordinates in [rad].

        """

        # Check range of input.
        coco.check_radial_coordinate(r)
        self.check_arm_index(arm_index)

        arm_param_vect = np.vstack(
            np.vectorize(self.arm_param.get, otypes=[np.ndarray])(arm_index)
        )

        phi = (
            arm_param_vect[:, 0] * np.log(r / arm_param_vect[:, 1])
            + arm_param_vect[:, 2]
        )

        return phi


class SpiralModelFK06(SpiralModelBase):
    """
    Spiral structure of Faucher-Giguère & Kaspi (2006) with the addition of the Local arm
    from Wainscoat et al. (1992).
    Spiral arm parameters from table 2 in Faucher-Giguère & Kaspi (2006) and table 1 in
    Wainscoat et al. (1992) assuming a Sun galactocentric distance R_sun = 8.5 kpc.
    The Local arm has a radial extension ~ 0.55 rad in the range [1.13, 1.68] rad
    (see Wainscoat et al. 1992).
    """

    def __init__(self):
        # Parameters of the model, values from Table 2 in Faucher-Giguère & Kaspi (2006) and
        # table 1 in Wainscoat et al. (1992). Respectively winding constant k [rad], the inner
        # radius r0 [kpc] and the inner angle phi0 [rad]. The phi0 values in Wainscoat et al.
        # (1992) are increased by pi/2 and reported in the range [0, 2pi].
        super().__init__()
        self.arm_param = {
            1: np.array([4.25, 3.48, 1.57]),  # Norma
            2: np.array([4.25, 3.48, 4.71]),  # Carina-Sagittarius
            3: np.array([4.89, 4.90, 4.09]),  # Perseus
            4: np.array([4.89, 4.90, 0.95]),  # Crux-Scutum
            5: np.array([4.57, 8.10, 1.13]),  # Local
        }
        # Probability of a star to be located in each of the spiral arms.
        # This is evaluated considering the stellar density and the length of the spiral arms
        # in the range of galactocentric radii where the local arm extends.
        self.arm_probability = [0.24615, 0.24615, 0.24615, 0.24615, 0.0154]
        # Galactocentric distances where the Local arm starts and ends [kpc].
        # These values are obtained by evaluating the r coordinates from the phi coordinates
        # specified in Wainscoat et al. (2014).
        self.local_r_min = 8.10
        self.local_r_max = 9.14


class SpiralModelYMW17(SpiralModelBase):
    """
    Spiral structure of Yao et al. (2017), see also Hou et al. (2014).
    Spiral arm parameters from table 1 in Yao et al. (2017) assuming a Sun
    galactocentric distance R_sun = 8.3 kpc. The Local arm has a radial extension
    ~ 1.05 rad in the range [0.87, 1.92] rad (see Hou et al. 2014).
    """

    def __init__(self):
        # Parameters of the model, values from Table 1 in Yao et al. (2017) re-adapted
        # to match the same logarithmic functional form used in Faucher-Giguère & Kaspi (2006).
        # Respectively winding constant k [rad], the inner radius r0 [kpc] and the inner angle
        # phi0 [rad].
        super().__init__()
        self.arm_param = {
            1: np.array([4.95, 3.35, 0.77]),  # Norma
            2: np.array([5.46, 3.56, 3.82]),  # Carina-Sagittarius
            3: np.array([5.77, 3.71, 2.09]),  # Perseus
            4: np.array([5.37, 3.67, 5.76]),  # Crux-Scutum
            5: np.array([20.67, 8.21, 0.96]),  # Local
        }
        # Probability of a star to be located in each of the spiral arms.
        # This is evaluated considering the stellar density and the length of the spiral arms
        # in the range of galactocentric radii where the local arm extends.
        self.arm_probability = [0.2455, 0.2455, 0.2455, 0.2455, 0.018]
        # Galactocentric distances where the Local arm starts and ends[kpc].
        # These values are obtained by evaluating the r coordinates from the phi coordinates
        # specified in Hou et al. (2014).
        self.local_r_min = 8.17
        self.local_r_max = 8.6
