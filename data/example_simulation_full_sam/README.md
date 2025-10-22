# Full simulation example with spiral arm model

This is an example of a full simulation (dynamical + magneto-rotational + detection) run with the `simulate_population_full.py` script and using the default parameters in the configuration file.

For this simulation, we specifically used a spiral arm model combined with a radial density model.
To this end, we have set the parameter `sample_edm` to `False` in the configuration file.

To run this example, we then use the following command:
```commandline
python mlpoppyns/simulator/simulate_population_full.py --save_dir data/example_simulation_full_sam
```