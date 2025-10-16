# Magneto-rotational + detection simulation example

This is an example of a magneto-rotational plus detection simulation run with the `simulate_population_magrot_det.py` script and changing the following parameters in the configuration file.
In order to activate the X-ray simulation:
cfg["simulation_xray"]: bool = True

We use the double log-normal model for the initial magnetic field distribution:
cfg["magnetic_field_model"]: str = "double_log-normal"

To run this example, we use the following command:
```commandline
python mlpoppyns/simulator/simulate_population_magrot_det.py --dyn_data data/example_simulation_dyn --save_dir data/example_simulation_magrot_det
```