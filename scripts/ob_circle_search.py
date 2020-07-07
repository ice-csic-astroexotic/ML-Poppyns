import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial

df = pd.read_csv("scripts/gaia_OB_stars.csv", sep=",", header=[0, 1])
values = df.to_numpy()
x = values[:, 0]
y = values[:, 1]

circles_x = np.random.random(10) * np.max(x)
circles_y = np.random.random(10) * np.max(y)
circles_r = np.random.random(10) * 3.0

print(x.shape)
print(y.shape)

fig, ax = plt.subplots()
plt.scatter(x, y)
for i in range(len(circles_r)):
    circle = plt.Circle((circles_x[i], circles_y[i]), circles_r[i], color="r")
    ax.add_artist(circle)
# plt.show()

points = np.c_[x.ravel(), y.ravel()]
tree = sp.spatial.KDTree(points)

results = []
for i in range(len(circles_x)):
    print("Searching circle {}".format(i))
    results.append(
        len(tree.query_ball_point([circles_x[i], circles_y[i]], circles_r[i]))
    )

print(results)
