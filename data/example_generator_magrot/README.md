# generator example
This is an example of dataset generation from a set of 20 simulations run with the `simulate_population_magrot_det.py` script and using the default parameters in the configuration file.

To run this example we use the following command:
```commandline
python pypopsyn/generator/generate_dataset_surveys.py --data data/example_simulation_helper_magrot --save_dir data/example_generator_magrot --resolution_dyn 32 --resolution_ppdot 32
```