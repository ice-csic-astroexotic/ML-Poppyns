#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Gaia Exploration script.

    Running the code:

        python3 gaia_exploration.py

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) 2020 MAGNESIA (ICE-CSIC)

"""

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astroquery.gaia import Gaia
from gala.units import galactic
from scipy.optimize import curve_fit

# Number of stars to query from the database.
STAR_NUMBER = 100

# SQL query for the database.
query_text = """SELECT TOP {} ra, dec, parallax, pmra, pmdec, radial_velocity
FROM gaiadr2.gaia_source
WHERE parallax > 0 AND
    radial_velocity IS NOT NULL
ORDER BY random_index
""".format(
    STAR_NUMBER
)

# The following lines will launch a job with the SQL query, get its resulting
# table and save it in FITS format. If you already have a table, just comment
# out those lines.
job = Gaia.launch_job(query_text)
gaia_data = job.get_results()
gaia_data.write("gaia_data.fits")

# Load the FITS data table from the SQL query.
gaia_data = QTable.read("gaia_data.fits")

# We want to convert the coordinate position and velocity data from heliocentric,
# spherical values to galactocentric cartesian ones. To do so with Astropy, we
# first have to create a SkyCoord object with the GAIA data we downloaded.

dist = coord.Distance(parallax=u.Quantity(gaia_data["parallax"]))

c = coord.SkyCoord(
    ra=gaia_data["ra"],
    dec=gaia_data["dec"],
    distance=dist,
    pm_ra_cosdec=gaia_data["pmra"],
    pm_dec=gaia_data["pmdec"],
    radial_velocity=gaia_data["radial_velocity"],
)

# Now that we have the SkyCoord object, we can transform them to any coordinate
# system. We transform it to the built-in galactocentric frame.
galcen = c.transform_to(
    coord.Galactocentric(z_sun=0 * u.pc, galcen_distance=8.1 * u.kpc)
)

print(galcen[:4])

# Just plotting a histogram of values here.
lp = np.linspace(-1000, 1000, 64)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.hist(galcen.v_z.value, bins=lp)
plt.xlabel("$v_z$ [{0:latex_inline}]".format(galcen.v_z.unit))
plt.ylabel("N")
plt.xticks([-1000, -500, 0, 500, 1000])
plt.grid()
plt.savefig("histogram.png")
plt.show()
