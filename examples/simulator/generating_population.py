import os

import hydra
import pandas as pd

import pypopsyn.simulator.configuration as configuration
import pypopsyn.simulator.initial_population as ipop


@hydra.main()
def generate_population(cfg) -> None:

    """ Generate an initial population.

    Args:

        cfg: configuration dictionary for the simulator.

    Returns:

        Nothing.

    """

    # Update simulator configuration with the provided parameters.
    configuration.update_configuration(cfg)

    # generating an initial neutron star population
    NS_population_initial = ipop.InitialNeutronStarPopulation()
    x_initial, y_initial, z_initial = NS_population_initial.position()
    (
        vp_x_initial,
        vp_y_initial,
        vp_z_initial,
    ) = NS_population_initial.proper_velocity()

    # adding the coordinates to a data frame for export
    df_initial = pd.DataFrame(
        {
            "x_initial": x_initial,
            "y_initial": y_initial,
            "z_initial": z_initial,
            "vp_x_initial": vp_x_initial,
            "vp_y_initial": vp_y_initial,
            "vp_z_initial": vp_z_initial,
        }
    )

    df_initial.columns = pd.MultiIndex.from_tuples(
        zip(
            df_initial.columns,
            ["[kpc]", "[kpc]", "[kpc]", "[km / s]", "[km / s]", "[km / s]"],
        )
    )

    print(os.getcwd())
    df_initial.to_csv("initial_population.txt", index=False, header=True)


if __name__ == "__main__":

    generate_population()
