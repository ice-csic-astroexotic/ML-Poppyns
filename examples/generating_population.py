import pandas as pd

import pypopsyn.simulator.initial_population as ipop

# generating an initial neutron star population

NS_population_initial = ipop.InitialNeutronStarPopulation()
x_initial, y_initial, z_initial = NS_population_initial.position()


# adding the coordinates to a data frame for export
df_initial = pd.DataFrame(
    {"x_initial": x_initial, "y_initial": y_initial, "z_initial": z_initial}
)

df_initial.columns = pd.MultiIndex.from_tuples(
    zip(df_initial.columns, ["[kpc]", "[kpc]", "[kpc]"])
)

df_initial.to_csv(
    "./examples/data/initial_population.txt", index=False, header=True
)
