# Generator ATNF example

This is an example of dataset generation from the ATNF Pulsar Catalogue.

To run this example, we use the following command:
```commandline
python mlpoppyns/generator/generate_observed_data.py --path_atnf data/observations/atnf_full_nobinary_24-09-2024_with_errors.csv --path_meerkat data/observations/meerkat_tpa_posselt_2023.csv --save_dir data/example_generator_observed --resolution_dyn 32 --resolution_ppdot 32
```