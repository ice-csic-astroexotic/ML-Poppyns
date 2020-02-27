"""
Evolving in time an initial population of neutron stars in the Milky Way. For now
only the dynamical evolution in the galactic potential is considered.
"""

import numpy as np
import pandas as pd

import pypopsyn.simulator.dynamical_evolution as dyn
import pypopsyn.simulator.initial_population as ipop

# upload data from the initial population simulation
data = pd.read_csv("./examples/data/initial_population.txt")
data = data[1:]

t_age = pd.to_numeric(data["age"]).values
r_initial = pd.to_numeric(data["r_initial"]).values
phi_initial = pd.to_numeric(data["phi_initial"]).values
z_initial = pd.to_numeric(data["z_initial"]).values
v_r_initial = pd.to_numeric(data["v_r_initial"]).values
v_phi_initial = pd.to_numeric(data["v_phi_initial"]).values
v_z_initial = pd.to_numeric(data["v_z_initial"]).values

omega_initial = v_phi_initial / r_initial

NS_number = np.len(t_age)

# define the initial conditions
initial_cond = np.array(
    [
        r_initial,
        phi_initial,
        z_initial,
        v_r_initial,
        omega_initial,
        v_z_initial,
    ]
).T

# evolve the positions and velocities of the neutron stars in time
final_population = dyn.dynamical_evolution(
    NS_number, initial_cond, t_age, time_step=1.0e4
)

r_final = final_population[:, 0]
phi_final = final_population[:, 1]
x_final = final_population[:, 2]
y_final = final_population[:, 3]
z_final = final_population[:, 4]
v_r_final = final_population[:, 5]
v_phi_final = final_population[:, 6]
v_z_final = final_population[:, 7]

# adding the coordinates to a data frame for export
df_final = pd.DataFrame(
    {
        "r_final": r_final,
        "phi_final": phi_final,
        "x_final": x_final,
        "y_final": y_final,
        "z_final": z_final,
        "v_r_final": v_r_final,
        "v_phi_final": v_phi_final,
        "v_z_final": v_z_final,
    }
)

df_final.columns = pd.MultiIndex.from_tuples(
    zip(
        df_final.columns,
        [
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

df_final.to_csv(
    "./examples/data/final_population.txt", index=False, header=True
)
