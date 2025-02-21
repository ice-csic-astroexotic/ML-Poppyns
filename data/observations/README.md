# Downloading ATNF Pulsar Catalogue

When downloading information from the [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/), 
we selected the following parameters:

JName, PMRA, PMDec, PX, PosEpoch, Gl, Gb, RaJD, DecJD, P0, P1, DM, W50, W10, Tau_sc, S400, S1400, S2000, Dist, Dist_DM, 
ZZ, XX, YY, Assoc, Survey, Type.

To exclude neutron stars in binaries, we use the filter `!type(BINARY)` in the `Condition` field.
We choose the output style to be `Short csv without errors` and replace the default null value `*` with `NAN`.

Finally, we click on the `TABLE` button at the end of the web page. 
This will produce a table that can be copied and pasted into a text editor and saved in `.csv` format.

Note that we have updated the spin periods of pulsars listed in Table 1 in [Song et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4562S/abstract) 
by hand because version 2.4.0 of the ATNF Catalogue used for this study does not contain the true pulsar
spin periods but instead the harmonics.

# Downloading TPA MeerKAT data

To download the TPA MeerKAT data, we refer to the work by [Posselt et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4582P/abstract). 
The data can be downloaded [here](https://academic.oup.com/mnras/article/520/3/4582/7049638#supplementary-data).

In particular, the table containing the flux information is given in the file `Census_Table5.csv`.

# Thermally emitting neutron stars

This catalog contains neutron star with a significant thermal emission component in their X-ray spectra. 
The data were collected and analyzed by Alessio Marino, Alice Borghese and Francesco Coti Zelati. 

The neutron stars are divided in classes: central compact objects (CCO), magnetars and 
rotational powered pulsars (RPP).
For the rotation powered pulsars the reported fluxes only come from the black-body component (the non-thermal component 
of the pulsar wind nebula has been removed as probably it is due to synchro-curvature emission mechanism). 
For the magnetars the reported fluxes include the non-thermal part as it is probably due to the resonant scattering in 
the magnetosphere of thermal photons coming from the hot surface. 

For the assoc. we indicate if the neutron star is associated with a pulsar wind nebula (PWN), has a radio counterpart 
and if it belongs to the Large Magellanic Cloud (LMC) or the small magellanic cloud (SMC).

