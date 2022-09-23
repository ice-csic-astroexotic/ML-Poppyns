*****************
Simulator Example
*****************

The simulation of a Galactic population of neutron stars can be performed by using different ways.

1) Simulate the entire population all together, which means initializing the entire population formed by N neutron stars from some initial conditions, evolve it in time and finally apply the survey models to select the detected neutron stars.
   For this approach you have to run the script :code:`examples/simulator/simulate_population_full.py`.
   This is usually fast since the simulator is optimized to work with multi-dimensional data thanks to the :code:`numpy` library.
   For example to simulate :math:`10^5` neutron stars with a maximum age of :math:`10^7` years the computational time is around 5 min.
   This approach is useful if you want to compare the simulated detected population with the entire simulated "unobservable" population and study the effect of the survey biases.
   This is possible since the information on the total simulated population is preserved and saved.
   The downside of this approach is that you need to assume a priori a neutron star birth rate, i.e. choose the maximum age and the total number of simulated neutron stars.

2) Simulate first the dynamical evolution of a huge number of neutron stars (without specifying the birth rate) by using the script :code:`examples/simulator/simulate_population_dyn.py`.
   The output of the dynamical evolution can be then used as a database for running the other steps of the simulations.
   Through the script :code:`examples/simulator/simulate_population_magrot_det.py` you then select stars from the dynamically evolved database according to the sky coverage of a given survey, evolve their properties in time and finally establish if they are detected or not by the surveys.
   New stars are selected and evolved until the desired number of detected sources is reached (for example to match the number of detected pulsars in a given survey in the ATNF catalog).
   This approach allows to find a posteriori the birth rate of the neutron stars by looking at the total number of neutron stars that have been created for a specified evolution time to reach the desired number of detections.

3) Initialize and evolve the neutron stars one-by-one until a number N of neutron stars has been simulated.
   For this third approach you have to run the script :code:`examples/simulator/simulate_population_1by1.py`.
   This approach is slower since it does not exploit the vectorization power of :code:`numpy`.
   For now only the dynamical and magneto-rotational evolution are available for this approach but not the survey selection.

Detailed information about the individual scripts can be obtained by issue the :code:`--h` argument, e.g., running:

.. code-block:: bash

  python examples/simulator/simulate_population_full.py --h

To simulate populations with different initial parameters, the user can directly modify the simulator configuration in :code:`pypopsyn/simulator/configuration.py` or alternatively for a more programmatic way a JSON dictionary containing configuration overrides for the simulation parameters can be provided as a command line argument to this script:

.. code-block:: bash

  python examples/simulator/simulate_population_full.py --output_dir simulated_data --parameter_override parameter_override.json

For example you can set the number of neutron stars to simulate, the models for the kick velocity distribution, the structure of the Galactic spiral arms, the number of spiral arms to include (5 or 4 depending if you want to include or not the Local arm) and several other parameters.

This will generate a new folder :code:`simulated_data` if it does not exist, in which the simulation results will be saved: the initial population in compressed binary format `initial_population.pkl.gz`, the final population in the same format `final_population.pkl.gz`, the profiles for the simulation if enabled and the dictionary containing the configuration parameters in `configuration.json` for reproducibility.

If the user opts to save the full evolutionary output for the dynamical and/or the magneto-rotational evolution by setting :code:`cfg["save_dyn_evolution"]` or :code:`cfg["save_magrot_evolution"]` to :code:`True` in the configuration file, a JSON file with the full time-stamped parameter evolution is also generated.
Note that since evolving the full population and saving the entire output requires a big computational cost and storage space, this feature should be enabled only for testing purposes when running the simulation on a reduced number of stars.
For example, to evolve and save both the full dynamical and magneto-rotational evolution for :math:`10^4` stars with a maximum age of :math:`10^7` years, the computation takes around 2 minutes and the JSON files containing the evolution outputs have a size of around 1 GB each.



If you instead want to run separately the dynamical evolution to create a database and then perform the rest of the simulation you can run the following scripts:

.. code-block:: bash

  python examples/simulator/simulate_population_dyn.py --output_dir dyn_database

This will create a population of neutron stars according to the initial conditions specified in the :code:`pypopsyn/simulator/configuration.py` and evolve it in time dynamically.
The output is saved in the specified output folder and as explained above it consists of the `initial_population.pkl.gz`, the final population in the same format `final_population.pkl.gz`, the profiles for the simulation if enabled and the dictionary containing the configuration parameters in `configuration.json` for reproducibility.
The user should ensure that the number of neutron stars evolved in this way is high enough to allow a proper determination of the birth rate in the following steps.
A safe number of neutron stars should be 30 per century, which is around 10 times the average core-collapse supernova rate in our Galaxy.

After the creation of the dynamical database you can run the script:

.. code-block:: bash

  python examples/simulator/simulate_population_magrot_det.py --dyn_data dyn_database --output_dir simulated_data

This script performs the following steps in a loop until the specified number of detections for each survey is reached:

1) It randomly samples batches of pulsars from the specified dynamical database.
   At the moment the batch size is set to 100000 and has been optimized for matching the radio observations with the Parkes telescope.
   This means that the dynamical database size should be bigger than this batch size, i.e., it should contain at least 100 times the neutron star number specified by the batch size.
   Simulating neutron stars in batches helps to speed up the simulation by evolving simultaneously an array of pulsars.

2) It selects pulsars that fall into the sky coverage of the surveys and are not futher away than 35 kpc from the Sun and evolves their magnetic field, spin period and inclination angle.
   This pre-selection allows to not waste computational resources on pulsars that have no chance to be detected.

3) The emission geometry is modeled so that only pulsars emitting towards the Earth can be selected.
   A luminosity is associated with these pulsars and their flux is computed.

4) It finally applies the modeled surveys to select the pulsars that are detected according to the limiting flux of each survey.

You can specify the number of pulsars you want to detect for each survey in the configuration file.
The simulation keeps track of the total number of created pulsars so that an estimate of the birth rate can be made a posteriori by knowing the maximum neutron star age that has been used for the simulation.
If the simulated birth rate exceeds a limiting value of 5 neutron stars per century, the simulation is stopped and a flag that warns about the excess in the birth rate is saved in the output configuration file.
In this case the output of the simulation will consist of separate files containing the parameters of the detected pulsars for each survey, a `profile.json` file and a `configuration.json` file containing the entire set of parameters used to simulate the magneto-rotational evolution and the detection models.

Simulation with Julia
#####################

The :code:`examples/simulator_julia` directory contains a version of the simulator scripts that uses Julia libraries
to perform the dynamical evolution of the population. The scripts work exactly in the same way described above.
However, in order to launch them it is necessary to run a different command:
To run python code that calls Julia is as simple as replacing :code:`python` with :code:`python-jl` in execution.
For the full simulation, we run:

.. code-block:: bash

  python-jl examples/simulator/simulate_population_full_julia.py --output_dir simulated_data

For the dynamical simulation only:

.. code-block:: bash

  python-jl examples/simulator_julia/simulate_population_dyn_julia.py --output_dir dyn_database


Simulations with parameter sweep
################################

`NOTE: The parameter sweep simulation helper has to be modified to adapt with the current versions of the simulators. The following section applies to an older version of the simulator.`

If you want to generate a huge parameter sweep you can use the wrapper or helper script that allows the specification of parameters with two types of sampling, determined by the argument :code:`--sampling_type`.
If :code:`--sampling_type = grid` you should provide the parameters in a linear spacing format :code:`--parameter [low] [high] [steps]`:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir simulated_data --kick_model "km_exp" --vk_c 100.0 200.0 100 --sampling_type grid

This example will generate a sweep of :code:`100` uniformly spaced samples for the :code:`vk_c` parameter in the range :code:`[100.0, 200.0]` using the :code:`km_exp` kick model and the results will be dumped in the specified :code:`simulated_data` folder.
If :code:`--sampling_type = random` you should provide the parameter ranges in the format :code:`--parameter [low] [high]` and specify the :code:`--sampling_size` argument which sets the number of values to draw from a uniform distribution for each parameter.

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir simulated_data --kick_model "km_exp" --vk_c 100.0 200.0 --sampling_type random --sampling_size 100

This example will generate a sweep of :code:`100` randomly drawn samples for the :code:`vk_c` parameter in the range :code:`[100.0, 200.0]` using the :code:`km_exp` kick model and the results will be dumped in the specified :code:`simulated_data` folder.


The parameters that can be swept with the simulation helper are :code:`vk_c` for the exponential kick velocity model  :code:`km_exp`, :code:`sigma_k` for the Maxwell kick velocity model :code:`km_maxwell`, :code:`h_c` for the galactic height distribution model of birth places, :code:`P_initial_mean` and :code:`P_initial_sigma` for the birth spin period distribution and :code:`B_initial_log10_mean` and :code:`B_initial_log10_sigma` for the birth magnetic field distribution.

You can also sweep over more than one parameter by running a script like:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir simulated_data --kick_model "km_exp" --vk_c 100.0 200.0 100 --h_c 0.01 2 10 --sampling_type grid

In this way a population is simulated for each combination of values of :code:`vk_c` and :code:`h_c`, i.e., the above case corresponds to :code:`100 x 10 = 1000` simulations.
The sweeper will generate a folder :code:`simulated_data` which will contain a folder for each simulation (parameter combination) named with an identifier number, i.e, :code:`000000`, :code:`000001`, :code:`000002` and so on.

Here it is important to distinguish between two types of arguments for the helper: options and parameters. Options are arguments which specify procedures for the simulation and they cannot be swept in a range since we assume they are not going to be outputs that will need to be predicted, e.g., the kick model. Parameters are values that we expect to use as ground truth for the learning system and therefore are to be predicted so they can be swept in a range to generate a dataset, e.g., :code:`vk_c`.

The script automatically checks the compatibility of the present parameters for the selected options, e.g., :code:`vk_c` cannot be specified if :code:`km_maxell` has been chosen as kick model. This is done using the dictionary :code:`examples/simulator/config_sweeper.json` which specifies a list of exclusive parameters for each option.

Visualize simulation results
############################

To visualize the outcome of a given simulation a number of jupyter notebooks are provided.
In particular to check the output of a full simulation run through the script :code:`examples/simulator/simulate_population_full.py` you can use the following ones:

1) :code:`initial_population_plots.ipynb` plots the initial parameter distributions of the population, by using the information stored in an `initial_population.pkl.gz` file.

2) :code:`final_population_plots.ipynb` plots the outcome of the simulation after the dynamical and magneto-rotational evolution and the detection, by using the information stored in a in a `final_population.pkl.gz` file.

3) To visualize and compare the initial and final parameter distributions for a simulation you can use the :code:`initial_final_comparison.ipynb` notebook.

4) To check the dynamical properties only, a comparison between the synthetic and the real observed pulsar population is performed in :code:`dynamics_sim_vs_obs.ipynb`.

5) The full dynamical and magneto-rotational evolution can be visualized with the notebook :code:`evolution_plots.ipynb`.
   This requires that you have saved the full dynamical and magneto-rotational output from the simulation.

To check the output of a simulation run through the script :code:`examples/simulator/simulate_population_magrot_det.py` and compare it to observations you can use the :code:`survey_sim_vs_obs.ipynb` notebook.