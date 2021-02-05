*****************
Simulator Example
*****************

The :code:`examples/simulator/initialize_evolve_population.py` script is responsible for initializing the simulated population from some initial conditions parameters and evolve it in time. To simulate populations with different initial parameters, the user can directly modify the simulator configuration in :code:`pypopsyn/simulator/configuration.py` or alternatively for a more programmatic way a JSON dictionary containing configuration overrides for the simulation parameters can be provided as a command line argument to this script:

.. code-block:: bash

  python examples/simulator/initialize_evolve_population.py --output_dir simulated_data --parameter_override override.json

This will generate a new folder :code:`simulated_data` if it does not exist, in which the simulation results will be saved: the initial population in compressed binary format `initial_population.pkl.gz`, the final population in the same format `final_population.pkl.gz`, the profiles for the simulation if enabled and the dictionary containing the parameter overrides in `override.json` for reproducibility.

For more information about the simulation script, issue the :code:`--h` argument:

.. code-block:: bash

  python examples/simulator/initialize_evolve_population.py --h

If you want to generate a huge parameter sweep you can use the wrapper or helper script that allows the specification of parameters in a linear spacing format :code:`--parameter [low] [high] [steps]`:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir simulated_data --kick_model "km_exp" --vk_c 100.0 200.0 100

This example will generate a sweep of :code:`100` uniformly spaced samples for the :code:`vk_c` parameter in the range :code:`[100.0, 200.0]` using the :code:`km_exp` kick model and the results will be dumped in the specified :code:`simulated_data` folder.
The parameters that can be swept with the simulation helper are :code:`vk_c` for the exponential kick velocity model  :code:`km_exp`, :code:`sigma_k` for the Maxwell kick velocity model :code:`km_maxwell` and :code:`h_c` for the galactic height distribution model of birth places.

You can also sweep over more than one parameter by running a script like:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir simulated_data --kick_model "km_exp" --vk_c 100.0 200.0 100 --h_c 0.01 2 10

In this way a population is simulated for each combination of values of :code:`vk_c` and :code:`h_c`, i.e., the above case corresponds to :code:`100 x 10 = 1000` simulations.
The sweeper will generate a folder :code:`simulated_data` which will contain a folder for each simulation (parameter combination) named with an identifier number, i.e, :code:`000000`, :code:`000001`, :code:`000002` and so on.

Here it is important to distinguish between two types of arguments for the helper: options and parameters. Options are arguments which specify procedures for the simulation and they cannot be swept in a range since we assume they are not going to be outputs that will need to be predicted, e.g., the kick model. Parameters are values that we expect to use as ground truth for the learning system and therefore are to be predicted so they can be swept in a range to generate a dataset, e.g., :code:`vk_c`.

The script automatically checks the compatibility of the present parameters for the selected options, e.g., :code:`vk_c` cannot be specified if :code:`km_maxell` has been chosen as kick model. This is done using the dictionary :code:`examples/simulator/config_sweeper.json` which specifies a list of exclusive parameters for each option.

To visualize the outcome of a given population two jupyter notebooks are provided: the first one :code:`initial_population_plots.ipynb` plots the initial conditions of the simulation, the second one :code:`final_population_plots.ipynb` plots the outcome of the simulation after the dynamical evolution.