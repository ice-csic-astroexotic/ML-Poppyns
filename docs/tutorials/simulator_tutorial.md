# Simulating neutron star populations

We can perform the simulation of a Galactic population of neutron stars in different ways.

In the following, we outline four approaches.

## End-to-end evolution

The first approach consists of simulating the full population in one go. This means that we have to 
initialize the entire population formed of N neutron stars from some initial conditions, evolve it in time 
and finally apply the survey models to select those neutron stars that are detected.

For this approach, we run the script `pypopsyn/simulator/simulate_population_full.py`.
Detailed information about the arguments that we can pass to the script is obtained by issuing the `--help` 
argument, e.g., running:
```
python pypopsyn/simulator/simulate_population_full.py --help
```
The default input parameters for the simulation are specified in the `pypopsyn/simulator/config_simulator.py` file.
To simulate populations with different initial parameters, the user can directly modify the simulator configuration in 
`pypopsyn/simulator/config_simulator.py`.

Alternatively, we can provide a JSON dictionary containing configuration overrides for the various simulation parameters. 
We specify this as a command line argument to the simulator script as follows:
```
python pypopsyn/simulator/simulate_population_full.py --save_dir output/sim_full --parameter_override parameter_override.json
```
For example, we can set the number of neutron stars to simulate, the kick-velocity model, the parameters of the initial 
distributions of spin periods and magnetic fields, and several other parameters.
The above command will then generate a new directory `output/sim_full` if it does not exist, in which the simulation 
results will be saved. The output consists of the file `initial_population.pkl.gz` containing the initial neutron star 
properties, the final population in the same format, i.e., `final_population.pkl.gz`, a `.pkl.gz` file for each of the 
modelled surveys (containing the stars detected by that survey), the profiles for the simulation if enabled and a 
dictionary with the configuration parameters in `configuration.json` for reproducibility.

!!! info

    A single simulation in this simulation mode is usually fast as the simulator is optimized to work with multi-dimensional data thanks to the `numpy` library. For example, to simulate $10^5$ neutron stars with a maximum age of $10^7$ years, the computational time is around five minutes.

Simulating the full population is useful if you want to compare the simulated detected population with the entire simulated
"unobserved" population and study the effects of survey biases.
This is possible as the information on the total simulated population, i.e., its initial conditions and the final evolved 
state is preserved and saved.
However, the downside of this approach is that we need to assume a neutron star birth rate a priori, i.e., choose the 
maximum age and the total number of simulated neutron stars.

If the user opts to save the full evolutionary output for the dynamical and/or the magneto-rotational evolution by setting 
`cfg["save_dyn_evolution"]` or `cfg["save_magrot_evolution"]` to `True` in the configuration file, a JSON file with the 
full time-stamped parameter evolution is also generated.
Note that since evolving the full population and saving the entire output requires large computational costs and storage 
space, this feature should be enabled only for testing purposes when running the simulation on a reduced number of stars.
For example, to evolve and save both the full dynamical and magneto-rotational evolution for $10^4$ stars with a maximum 
age of $10^7$ years, the computation takes around 20 seconds and the JSON files containing the evolution outputs have a 
size of around 600 and 400 Mb each.

!!! example

    An example of this simulation mode is presented in detail in the tutorial `tutorials/tutorial_notebooks/01_simulator_full_tutorial.ipynb`. 

## Dynamical evolution


If we instead want to perform the dynamical evolution only, we run the following script:
```
python pypopsyn/simulator/simulate_population_dyn.py --save_dir output/sim_dyn
```
As for the case above, to change the initial parameters, the user can directly modify the simulator 
configuration in `pypopsyn/simulator/config_simulator.py` or alternatively parse a JSON file containing 
custom simulation parameters.

The above command will create a population of neutron stars according to the initial conditions specified in
`pypopsyn/simulator/config_simulator.py` and evolve it in time dynamically.
The output is saved in the specified output folder and consists of a file `final_pop_dyn.csv` containing the 
information on the final positions and velocities of neutron stars in the Galaxy, the profiles for the 
simulation if enabled and a dictionary of configuration parameters in `configuration.json` for 
reproducibility.

This simulation approach is useful if the user wants to create a database of dynamically evolved neutron stars and use 
it to separately run subsequent simulation steps, i.e., perform the magneto-rotational evolution and apply observational 
filters (see below). In this case, we need to ensure that the number of neutron stars evolved dynamically is 
sufficiently large to allow for a proper determination of the birth rate in the subsequent steps. A safe assumption is 
30 neutron stars are born per century, which is around 10 times the average core-collapse supernova rate in our Galaxy.

!!! example

    An example of this simulation mode is presented in detail in the tutorial `tutorials/tutorial_notebooks/02_simulator_dyn_tutorial.ipynb`.

## Magneto-rotational evolution and detection filters

After simulating the dynamical evolution of a large number of neutron stars by using the script 
`pypopsyn/simulator/simulate_population_dyn.py`, the corresponding database can be used as a basis for the subsequent 
steps of pulsar population synthesis.

Using the module `pypopsyn/simulator/simulate_population_magrot_det.py`, we can select stars from the 
dynamically evolved database according to the sky coverage of a given survey, evolve their properties in 
time and finally establish if these sources are detected by a given survey or not.

To do so, we execute the command
```
python pypopsyn/simulator/simulate_population_magrot_det.py --dyn_data dyn_database --save_dir output/sim_magrot_det
```
As for the cases above, to change the initial parameters, the user can directly modify the simulator 
configuration in `pypopsyn/simulator/config_simulator.py` or alternatively parse a JSON file containing 
custom parameters for the simulation.

In this simulation mode, new stars are sampled from the database in batches and evolved until the desired 
number of detected sources is reached. Note that we can specify the number of pulsars that we want to detect
for each survey in the configuration file. For example, we might want to match the number of detected 
pulsars in a given survey in the [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/). 
To this end, the script performs the following steps in a loop until a certain number of detections for each 
survey is reached:

1. We first randomly sample batches of pulsars from our dynamical database. The batch size is set to 100,000 and has 
    been optimized to match radio pulsar detections with Murriyang, the Parkes Radio Telescope. Note that the dynamical 
    database should be significantly bigger than this batch size, i.e., it should contain at least 100 times the batch 
    size. Simulating neutron stars in batches helps to speed up the simulation by evolving an array of pulsars
    simultaneously.

2. Next, we select pulsars that fall into the sky coverage of the surveys and are not further away than 35 kpc from 
    the Sun. For those objects, we then evolve the magnetic fields, spin periods and inclination angles. This 
    pre-selection allows us to not waste computational resources on pulsars that have no chance to be detected.

3. Then, we model the emission geometry and only select those pulsars that are beamed towards the Earth. A luminosity is
    associated with these pulsars and their flux is computed.

4. Finally, we apply the surveys' sensitivity thresholds to select those pulsars that are detected according to the 
    limiting flux of each survey.

A posteriori, this simulation mode allows us to determine the birth rate of our synthetic neutron star population by 
looking at the total number of neutron stars created over a specified evolution time to reach the desired number of 
detections. To avoid wasting computational resources on unrealistic parameter regimes, we do not evolve populations 
whose birth rate exceeds a limiting value of 5 neutron stars per century. In these cases, the simulation is stopped 
and a flag that warns about the excess in the birth rate is saved into the output configuration file.

Overall, in this mode, the output of the simulation consists of separate files containing the properties of the 
detected pulsars for each survey, a `profile.json` file and a `configuration.json` file containing the entire set of 
parameters used to simulate the magneto-rotational evolution and the detection models.

!!! example

    An example of this simulation mode is presented in detail in the tutorial `tutorials/tutorial_notebooks/03_simulator_magrot_det_tutorial.ipynb`.

## Simulations with parameter sweep

If we want to run simulations for a large number of parameter combinations, we can use the helper script 
`run_simulation_set.py` in the `utilities/simulation_helper` folder. This script allows us to run multiple simulations 
with different input parameters in an automated manner.

In this mode, we can choose the type of simulation we want to run (i.e., `simulate_population_dyn.py`, 
`simulate_population_magrot_det.py` or `simulate_population_full.py`) via the `--simulator_type` argument. 
In addition, the number of cores used to run the simulation in parallel is specified through `--processes`.
Finally, we can choose between two sampling approaches to determine the relevant simulation parameters through the 
argument `--sampling_type`. 

Specifically, if `--sampling_type = grid`, a regular multi-dimensional grid is created, and we need to 
provide the parameters in a linear spacing format `--parameter [low] [high] [steps]`. For example,
```
python utilities/simulation_helper/run_simulation_set.py --simulator_type simulate_population_dyn --save_dir output/sim_helper --vk_c 100.0 200.0 20 --sampling_type grid
```
This command will run the dynamical simulation only and generate a sweep of `20` uniformly spaced samples 
for the `vk_c` parameter in the range `[100.0, 200.0]`. As this parameter is related to the exponential 
kick model `km_exp`, the script will first check if this model is properly set in the `config_simulator.py` 
file. If not, an error will be thrown.

On the other hand, if `--sampling_type = random`, the parameters are sampled uniformly at random within the 
provided limits. In this case, we need to provide the parameter ranges in the format `--parameter 
[low] [high]` and specify the `--sampling_size` argument, which sets the number of values drawn from a 
uniform distribution for each parameter. For example,
```
python utilities/simulation_helper/run_simulation_set.py --simulator_type simulate_population_dyn --save_dir output/sim_helper --vk_c 100.0 200.0 --sampling_type random --sampling_size 20
```
This will generate a sweep of `20` randomly drawn samples for the `vk_c` parameter in the range `[100.0, 200.0]`.

In the following, we list the parameters that can be swept with the `run_simulation_set.py` script.

For the dynamical evolution:

* `vk_c` for the exponential kick-velocity model `km_exp`;
* `sigma_k` for the Maxwell kick-velocity model `km_maxwell`;
* `h_c` for the Galactic scale height in the exponential model of birthplaces.

For the magneto-rotational evolution:

* `P_initial_mean` and `P_initial_sigma` for the birth spin periods in the `normal` distribution model;
* `P_initial_log10_mean` and `P_initial_log10_sigma` for the birth spin periods in the `log-normal` distribution model;
* `B_initial_log10_mean` and `B_initial_log10_sigma` for the birth magnetic fields in the log-normal distribution model;
* `a_late` for the power-law index describing the late-time decay of the magnetic field.

For the radio emission:

* `L_radio_log10_mean` and `epsilon_L` for the power-law model describing the pulsars' radio luminosity.

!!! note

    It is important to distinguish between two types of arguments for the `run_simulation_set.py` script: options 
    and parameters. Options are arguments which specify procedures for the simulation helper, and they cannot be 
    swept across a range, e.g., the dynamical database or the simulator type. By definition, these are not outputs 
    that we want to predict with our subsequent machine-learning framework. On the other hand, parameters are values
    that we expect to use as ground truths for the learning pipeline and want to predict. Hence, these can be swept 
    in a range to generate a set of training simulations.

    The script automatically checks for the compatibility of the given parameters and the selected options, e.g., 
    `vk_c` cannot be specified if `km_maxwell` has been chosen as the kick model in the `config_simulator.py`. 
    The dictionary `utilities/config_sweeper.json` specifies a list of required parameters for each option.

Finally, we can also sweep over more than one parameter. For example, let us assume that we want to simulate 20 
populations of neutron stars varying the parameters `P_initial_log10_mean` in the range -1.5 to -0.3 and the parameter 
`B_initial_log10_mean` in the range 12 to 14 using the `simulate_population_magrot_det.py` script and a previously 
simulated dynamical database saved in `data/example_simulation_dyn`. We can then run:
```
  python utilities/simulation_helper/run_simulation_set.py --simulator_type simulate_population_magrot_det --save_dir output/sim_helper --dyn_data data/example_simulation_dyn --P_initial_log10_mean -1.5 -0.3 --B_initial_log10_mean 12 14 --sampling_type random --sampling_size 20
```
In this way, a population is simulated for each of the 20 pairs of random values of `P_initial_log10_mean` and 
`B_initial_log10_mean`. The sweep will generate a directory `output/sim_helper`, which will contain a folder for each 
simulation (parameter combination) named with an identifier, i.e., `000000`, `000001`, `000002` and so on.

If instead `--sampling_type grid`, we need to specify the command as follows:
```
  python utilities/simulation_helper/run_simulation_set.py --simulator_type simulate_population_magrot_det --save_dir output/sim_helper --dyn_data data/example_simulation_dyn --P_initial_log10_mean -1.5 -0.3 10 --B_initial_log10_mean 12 14 10 --sampling_type grid
```
This simulates a population for each combination of values of `P_initial_log10_mean` and `B_initial_log10_mean`, i.e., the 
above case corresponds to `10 x 10 = 100` simulations. Again, each simulation will be saved in a separate directory 
`000000`, `000001`, `000002` and so on.


!!! example

    An example of this simulation mode is presented in detail in the tutorial `tutorials/tutorial_notebooks/04_simulation_helper_tutorial.ipynb`.



