# Simulation helper example
This is an example of a set of 20 simulations run with the `run_simulation_set.py` script for the `simulate_population_magrot_det.py` script.
We varied the parameter `P_initial_log10_mean` in the range -1.5 -0.3 and the parameter `B_initial_log10_mean` in the range 12 14.

To run this example we use the following command:
```commandline
python utilities/simulation_helper/run_simulation_set.py --simulator_type simulate_population_magrot_det --dyn_data data/example_simulation_dyn --save_dir data/example_simulation_helper_magrot --sampling_type random --sampling_size 20 --processes 20 --P_initial_log10_mean -1.5 -0.3 --B_initial_log10_mean 12 14
```