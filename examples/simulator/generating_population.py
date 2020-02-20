import pandas as pd

import pypopsyn.simulator.initial_population as ipop

# generating an initial neutron star population

NS_population_initial = ipop.InitialNeutronStarPopulation()

# generating ages
age = NS_population_initial.age()

# generating initial positions
(
    r_initial,
    phi_initial,
    x_initial,
    y_initial,
    z_initial,
) = NS_population_initial.position(t_age=age)

# generating initial velocities summing the proper velocities to the orbital velocities
(vp_r, vp_phi, vp_z,) = NS_population_initial.proper_velocity()

v_orb = NS_population_initial.orbital_velocity(r_initial, z_initial)

v_r_initial = vp_r
v_phi_initial = vp_phi + v_orb
v_z_initial = vp_z

# adding the coordinates to a data frame for export
df_initial = pd.DataFrame(
    {
        "age": age,
        "r_initial": r_initial,
        "phi_initial": phi_initial,
        "x_initial": x_initial,
        "y_initial": y_initial,
        "z_initial": z_initial,
        "v_r_initial": v_r_initial,
        "v_phi_initial": v_phi_initial,
        "v_z_initial": v_z_initial,
    }
)

df_initial.columns = pd.MultiIndex.from_tuples(
    zip(
        df_initial.columns,
        [
            "[yr]",
            "[kpc]",
            "[rad]",
            "[kpc]",
            "[kpc]",
            "[kpc]",
            "[kpc / yr]",
            "[kpc /yr]",
            "[kpc / yr]",
        ],
    )
)

df_initial.to_csv(
    "./examples/data/initial_population.txt", index=False, header=True
)
