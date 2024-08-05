# Full simulation example with spiral arm model

This is an example of a full simulation (dynamical + magneto-rotational + detection) run with the `simulate_population_full.py` script and using the default parameters in the configuration file.

For this simulation we used the spiral arm model and galactic structure defined in the `pypopsyn/simulator/initial_population_sam.py` module.
For this to work one has to modify the `pypopsyn/simulator/simulate_population_full.py` by importing the modules:
```commandline
import pypopsyn.simulator.initial_population_sam as ipop
import pypopsyn.simulator.stellar_dynamics.spiral_model as sm
```
and specify the spiral arm model when initializing the initial positions by changing line 191 in the code to 
```
(
    r_initial,
    phi_initial,
    z_initial,
) = NS_population_initial.position(t_age=age, spiral_model=sm.spiral_model)
```
This is needed because the default model relies on the electron density to populate the Galaxy with neutron stars.

To run this example, we then use the following command:
```commandline
python pypopsyn/simulator/simulate_population_full.py --save_dir data/example_simulation_full_sam
```