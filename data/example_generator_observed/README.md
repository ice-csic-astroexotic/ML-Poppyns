# Generator ATNF example

This is an example of dataset generation from the ATNF Pulsar Catalogue, Meerkat radio fluxes and thermally emitting neutron star catalog.
For the thermally emitting neutron stars, we apply the filter to only consider young magnetars (age < 2 kyr) and XDINSs (distance < 500 pc and age > $10^5$ yr).

To run this example, we use the following command:
```commandline
python mlpoppyns/generator/generate_observed_data.py --path_atnf data/observations/atnf_full_nobinary_24-09-2024_with_errors.csv --path_meerkat data/observations/meerkat_tpa_posselt_2023.csv --path_xray data/observations/thermal_NS_05-11-2024.csv --save_dir data/example_generator_observed --resolution_dyn_radio 32 --resolution_ppdot_radio 32 --resolution_dyn_xray 32 --resolution_ppdot_xray 32 --filter_young_xdins
```