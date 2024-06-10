# generator example
This is an example of dataset generation from a set of 20 simulations run with the `simulate_population_magrot_det.py` script and using the default parameters in the configuration file.

To run this example we use the following command:
```commandline
python pypopsyn/generator/generate_dataset_surveys.py --data data/example_simulation_helper_magrot --save_dir data/example_generator_magrot --resolution_dyn 32 --resolution_ppdot 32
```

In order to split the dataset and use 80% for training/validation and 20% for testing we use the following command:
```commandline
python pypopsyn/generator/dataset_splitter.py --dataset_path data/example_generator_magrot --valid_split 0.2
```
Note that this script generates a `dataset_train.csv`, a `dataset_valid.csv` and a `statistics_train.json`.
We renamed the `dataset_valid.csv` into `dataset_test.csv`.