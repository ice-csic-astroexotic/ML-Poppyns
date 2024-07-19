# Simulating neutron star populations

The simulation of a Galactic population of neutron stars can be performed by using different ways.

## Simulating a full population of neutron stars

The first approach consists in simulating the entire population all together, which means initializing the entire population formed by N neutron stars from some initial conditions, evolve it in time and finally apply the survey models to select the detected neutron stars.
For this approach you have to run the script `pypopsyn/simulator/simulate_population_full.py`.
Detailed information about the arguments to pass to the script can be obtained by issue the `--h` argument, e.g., running:
```
python pypopsyn/simulator/simulate_population_full.py --h
```
The default input parameters of the simulation are specified in the `pypopsyn/simulator/config_simulator.py` file.
To simulate populations with different initial parameters, the user can directly modify the simulator configuration in `pypopsyn/simulator/config_simulator.py` or alternatively for a more programmatic way a JSON dictionary containing configuration overrides for the simulation parameters can be provided as a command line argument to this script:
```
python pypopsyn/simulator/simulate_population_full.py --save_dir output/sim_full --parameter_override parameter_override.json
```
For example you can set the number of neutron stars to simulate, the kick-velocity model the parameters of the initial distribution of spin periods and magnetic fields and several other parameters.
This will generate a new directory `output/sim_full` if it does not exist, in which the simulation results will be saved.
The output consists of the file `initial_population.pkl.gz`containing the initial conditions, the final population in the same format `final_population.pkl.gz`, a `.pkl.gz` file for each one of the modelled surveys containing the population detected by that survey, the profiles for the simulation if enabled and the dictionary containing the configuration parameters in `configuration.json` for reproducibility.

This simulation is usually fast since the simulator is optimized to work with multi-dimensional data thanks to the `numpy` library.
For example to simulate $10^5$ neutron stars with a maximum age of $10^7$ years the computational time is around 5 min.
This approach is useful if you want to compare the simulated detected population with the entire simulated "unobservable" population and study the effect of the survey biases.
This is possible since the information on the total simulated population, i.e., its initial conditions and the final evolved state is preserved and saved.
The downside of this approach is that you need to assume a priori a neutron star birth rate, i.e. choose the maximum age and the total number of simulated neutron stars.

If the user opts to save the full evolutionary output for the dynamical and/or the magneto-rotational evolution by setting `cfg["save_dyn_evolution"]` or `cfg["save_magrot_evolution"]` to `True` in the configuration file, a JSON file with the full time-stamped parameter evolution is also generated.
Note that since evolving the full population and saving the entire output requires a big computational cost and storage space, this feature should be enabled only for testing purposes when running the simulation on a reduced number of stars.
For example, to evolve and save both the full dynamical and magneto-rotational evolution for $10^4$ stars with a maximum age of $10^7$ years, the computation takes around 20 seconds and the JSON files containing the evolution outputs have a size of around 600 and 400 Mb each.

To see a tutorial example for this simulator you can look at the notebook in `tutorials/tutorial_notebooks/simulator_full_tutorial.ipynb`. 

## Simulating the dynamical evolution of a population of neutron stars

If you instead want to run only the dynamical evolution you can run the following script:
```
python pypopsyn/simulator/simulate_population_dyn.py --save_dir output/sim_dyn
```
As for the case above, to change the initial parameters, the user can directly modify the simulator configuration in `pypopsyn/simulator/config_simulator.py` or alternatively parsing a JSON file containing custom parameters for the simulation.
This will create a population of neutron stars according to the initial conditions specified in the :code:`pypopsyn/simulator/config_simulator.py` and evolve it in time dynamically.
The output is saved in the specified output folder and it consists of a file `final_pop_dyn.csv` containing the information on the final positions and velocities of neutron stars in the Galaxy, the profiles for the simulation if enabled and the dictionary containing the configuration parameters in `configuration.json` for reproducibility.

This approach is useful if the user would like to create a database of dynamically evolved neutron stars and use it to run the other simulation steps afterwards, i.e., perform the magneto-rotational evolution and apply observational filters (see below).
In this case the user should ensure that the number of neutron stars evolved in this way is high enough to allow a proper determination of the birth rate in the following steps.
A safe number of neutron stars should be 30 per century, which is around 10 times the average core-collapse supernova rate in our Galaxy.

To see a tutorial example for this simulator you can look at the notebook in `tutorials/tutorial_notebooks/simulator_dyn_tutorial.ipynb`. 

## Simulating the magneto-rotational evolution and detection of a population of neutron stars

After simulating the dynamical evolution of a huge number of neutron stars by using the script `pypopsyn/simulator/simulate_population_dyn.py` the output of the dynamical evolution can be then used as a database for running the other steps of the simulations.
Through the script `pypopsyn/simulator/simulate_population_magrot_det.py` you can select stars from the dynamically evolved database according to the sky coverage of a given survey, evolve their properties in time and finally establish if they are detected or not by the surveys.
New stars are selected and evolved until the desired number of detected sources is reached (for example to match the number of detected pulsars in a given survey in the ATNF catalog).
This approach allows to find a posteriori the birth rate of the neutron stars by looking at the total number of neutron stars that have been created for a specified evolution time to reach the desired number of detections.

After the creation of the dynamical database you can run the script:
```
python pypopsyn/simulator/simulate_population_magrot_det.py --dyn_data dyn_database --save_dir output/sim_magrot_det
```
As for the cases above, to change the initial parameters, the user can directly modify the simulator configuration in `pypopsyn/simulator/config_simulator.py` or alternatively parsing a JSON file containing custom parameters for the simulation.

This script performs the following steps in a loop until the specified number of detections for each survey is reached:

1) It randomly samples batches of pulsars from the specified dynamical database.
   At the moment the batch size is set to 100000 and has been optimized for matching the radio observations with the Parkes telescope.
   This means that the dynamical database size should be bigger than this batch size, i.e., it should contain at least 100 times the neutron star number specified by the batch size.
   Simulating neutron stars in batches helps to speed up the simulation by evolving simultaneously an array of pulsars.

2) It selects pulsars that fall into the sky coverage of the surveys and are not further away than 35 kpc from the Sun and evolves their magnetic field, spin period and inclination angle.
   This pre-selection allows to not waste computational resources on pulsars that have no chance to be detected.

3) The emission geometry is modeled so that only pulsars emitting towards the Earth can be selected.
   A luminosity is associated with these pulsars and their flux is computed.

4) It finally applies the modeled surveys to select the pulsars that are detected according to the limiting flux of each survey.

You can specify the number of pulsars you want to detect for each survey in the configuration file.
The simulation keeps track of the total number of created pulsars so that an estimate of the birth rate can be made a posteriori by knowing the maximum neutron star age that has been used for the simulation.
If the simulated birth rate exceeds a limiting value of 5 neutron stars per century, the simulation is stopped and a flag that warns about the excess in the birth rate is saved in the output configuration file.
In this case the output of the simulation will consist of separate files containing the parameters of the detected pulsars for each survey, a `profile.json` file and a `configuration.json` file containing the entire set of parameters used to simulate the magneto-rotational evolution and the detection models.

To see a tutorial example for this simulator you can look at the notebook in `tutorials/tutorial_notebooks/simulator_magrot_det_tutorial.ipynb`. 





