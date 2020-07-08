import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial

df_grid = pd.read_csv("scripts/grid.csv", sep=",", header=[0, 1])
grid_values = df_grid.to_numpy()
circles_x = grid_values[:, 0]
circles_y = grid_values[:, 1]
circles_r = grid_values[:, 2]

df = pd.read_csv("scripts/gaia_OB_stars.csv", sep=",", header=[0, 1])
values = df.to_numpy()
stars_x = values[:, 0]
stars_y = values[:, 1]

points = np.c_[stars_x.ravel(), stars_y.ravel()]
tree = sp.spatial.KDTree(points)

results = []
for i, (x, y, radius) in enumerate(zip(circles_x, circles_y, circles_r)):
    print("Searching circle {}".format(i))
    results.append(len(tree.query_ball_point([x, y], radius)))

print(results)


fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.set_xlabel(r"$x$ [kpc]")
ax.set_ylabel(r"$y$ [kpc]")
ax.set_xlim([-10, 10])
ax.set_ylim([0, 15])
plt.scatter(stars_x, stars_y, s=0.001, color="teal")

cmap = matplotlib.cm.get_cmap("jet")
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(np.array(results)))

for i in range(0, len(circles_r)):
    circle = plt.Circle(
        (circles_x[i], circles_y[i]),
        circles_r[i],
        color=cmap(norm(results[i])),
        linewidth=0,
        rasterized=True,
        alpha=0.25,
    )
    ax.add_artist(circle)
    ax.annotate(
        str(results[i]),
        xy=(circles_x[i], circles_y[i]),
        fontsize=8,
        ha="center",
    )

sc = plt.scatter(
    circles_x,
    circles_y,
    s=0,
    c=results,
    cmap="jet",
    vmin=0.0,
    vmax=np.max(np.array(results)),
    facecolors="none",
)
cbar = plt.colorbar(sc)
cbar.set_label("Count", rotation=270, labelpad=10)

fig.savefig("circular_count.png")
