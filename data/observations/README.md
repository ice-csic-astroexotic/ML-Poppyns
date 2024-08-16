# Download ATNF catalog

To download the [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/) we selected the following parameters:
JName, PMRA, PMDec, PX, PosEpoch, Gl, Gb, RaJD, DecJD, P0, P1, DM, W50, W10, Tau_sc, S400, S1400, S2000, Dist, Dist_DM, ZZ, XX, YY, Assoc, Survey, Type.
To exclude neutron stars in binaries we use the filter `!type(BINARY)` in the `Condition` field. 
We choose the output style to be `Short csv without errors` and replace the default null value with `NAN`.
And finally click on the `TABLE` button at the end of the web page. This will produce a table that can be copied and pasted in a text editor and saved in `.csv` format.

# Download TPA Meerkat data

To download the TPA Meerkat data we refer to the work in [(Posselt et al., 2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4582P/abstract). 
The data can be downloaded [here](https://academic.oup.com/mnras/article/520/3/4582/7049638#supplementary-data).
In particular the table containing the flux information is in the file `Census_Table5.csv`.