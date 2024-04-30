"""
Model for computing the hydrogen column density for X-ray absorption.

For computing the N_H we provide two options:
1) Using the map and routines from the 3D N_H-tool by Doroshenko (2024),
available for download at https://zenodo.org/records/10779060.
2) Using the relation between N_H and DM found by He, Ng and Kaspi (2013).

Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os
import pathlib
import sys

import astropy.coordinates as c
import astropy.units as u
import healpy as hp
import numpy as np

import scripts.download_online_content as do
from pypopsyn.simulator.config_simulator import cfg

log = logging.getLogger(__name__)


def compute_NH(RA: np.ndarray, DEC: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Given an array of positions in equatorial coordinates and distances, return the corresponding line of sight N_H
    using Wilms et al. (2000) abundances in the given directions.
    For computing the N_H, we use the reddening map and routines from Doroshenko (2024),
    available for download at https://zenodo.org/records/10779060.

    Args:
        RA (np.ndarray): array of right ascension coordinates in deg.
        DEC (np.ndarray): array of declination coordinates in deg.
        d (np.ndarray): array of distances in kpc.

    Return:
        N_H (np.ndarray): array of hydrogen column densities in cm^(-2).
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Download and load the reddening (E(B-V)) map from Doroshenko (2024).
    # The .npz file contains some calibration parameters (see below), an array of 1075 distance bins from 0 to 25 kpc
    # from the Sun and a reddening map with shape (1075, 786432), where 1075 is the number of distance bins and 786432
    # is the number of pixels covering a spherical shell around the Sun at a given distance.
    # Each row of the map corresponds to a given sky coordinate, while each column corresponds to a given distance
    # from the Sun.
    map_path = pathlib.Path().joinpath(
        cfg["path_to_software"],
        "pypopsyn/simulator/interstellar_medium/ebv_map.npz",
    )

    try:
        if not os.path.exists(map_path):
            download_url = "https://zenodo.org/records/10779060/files/ebv_fin.npz?download=1"
            destination_path = map_path
            log.info(
                f"Downloading the reddening map from: {download_url}. "
                f"Note that this might take a few minutes as the map is 1.3 GB. "
                f"Once the download is completed, the map will be saved in pypopsyn/simulator/interstellar_medium/."
            )
            do.download_file(download_url, str(destination_path))

        # Load the reddening map.
        maps = np.load(map_path)["maps"].T

        # Load the distance bins.
        dbins = np.load(map_path)["radius"]

    except Exception as e:
        log.error(
            f"An error occurred: {str(e)}. "
            f"Remember to set the right absolute path_to_software in the configuration file."
        )

    # Load the calibration parameters.
    # - Rv is the extinction law parameter or the total-to-selective extinction ratio, and represents the ratio
    # of total extinction (Av) to selective extinction (E(B − V)). It tells us how much more extinction occurs
    # in the visual (V) band compared to the extinction in the blue (B) band.
    # - calib_avks is a conversion factor from visual extinction (Av) to infrared extinction (Aks).
    # - calib_nhag89 is a conversion factor from reddening (E(B − V)) to N_H and assumes solar abundances from
    # Anders & Grevesse (1989).
    # - calib_nhw00 is a conversion factor from reddening (E(B − V)) to N_H and assumes sub-solar abundances from
    # Wilms et al. (2000).
    (
        Rv,
        calib_avks_mean,
        calib_avks_std,
        calib_nhag89_mean,
        calib_nhag89_std,
        calib_nhw00_mean,
        calib_nhw00_std,
    ) = [
        np.load(map_path)["%s" % x]
        for x in [
            "Rv",
            "calib_avks_mean",
            "calib_avks_std",
            "calib_nhag89_mean",
            "calib_nhag89_std",
            "calib_nhw00_mean",
            "calib_nhw00_std",
        ]
    ]

    # Create an astropy.coordinates.SkyCoord object.
    pos = c.SkyCoord(RA * u.deg, DEC * u.deg, distance=d * u.kpc, frame="fk5")

    # Transform to Galactic coordinates and the distances in pc.
    gpos = pos.transform_to(c.Galactic)
    dist = np.array(gpos.distance.pc)

    # If distance exceeds 25 kpc, set it to the maximum distance in the map, i.e., 25 kpc.
    dist[dist > 25000] = np.array([dbins[-1]])

    # Find the pixels of the map corresponding to the given coordinates with healpy. This will select a row in the map.
    gpix = np.array(hp.ang2pix(256, gpos.l.deg, gpos.b.deg, lonlat=True))

    # Extract the indices corresponding to the given distances from the map.
    idx = np.minimum(dbins.searchsorted(dist), len(dbins) - 1)

    # Extract the reddening E(B-V) corresponding to the given pixel and distance for every star.
    ebv = maps[gpix, idx]

    # Convert the reddening to an N_H estimate for each neutron star by multiplying the value of the map by the
    # calibration factor (use Wilms et al. (2000) abundances by default).
    nh_conv_fac = calib_nhw00_mean[0]
    nh = ebv * nh_conv_fac
    N_H = np.array(nh * 10**21, dtype=float)

    return N_H


def compute_NH_from_DM(DM: np.ndarray) -> np.ndarray:
    """
    Given an array of dispersion measures, DM, estimate the corresponding line of sight N_H
    using the relation found by He, Ng and Kaspi (2013). This relation might underestimate the N_H
    estimate for large values of N_H (see He, Ng and Kaspi (2013) for more details).

    Args:
        DM (np.ndarray): array of dispersion measures in pc cm^-3.

    Return:
        N_H (np.ndarray): array of hydrogen column densities in cm^(-2).
    """

    N_H = np.array([0.3 * DM * 10**20], dtype=float)

    return N_H
