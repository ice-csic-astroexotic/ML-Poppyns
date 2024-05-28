# Full simulation example
This is an example of full simulation (dynamical + magneto-rotational + detection) run with the `simulate_population_full.py` script and using the default parameters in the configuration file.
For this simulation we used the spiral arm model and galactic structure defined in the `pypopsyn/simulator/initial_population.py` module.
For this to work one has to modify the `pypopsyn/simulator/simulate_population_full.py` by importing the module:
```commandline
import pypopsyn.simulator.initial_population
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
```
and specify the spiral arm model when initializing the initial positions by changing line 191 in the code with 
```
(
    r_initial,
    phi_initial,
    z_initial,
) = NS_population_initial.position(t_age=age, spiral_model=sm.spiral_model)
```

To run this example we then use the following command:
```commandline
python pypopsyn/simulator/simulate_population_full.py --output_dir data/example_simulation_full
```