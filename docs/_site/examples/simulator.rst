*****************
Simulator Example
*****************

The :code:`examples/simulator/initialize_evolve_population.py` script is responsible for initializing the simulated population from some initial conditions parameters and evolve it in time. To simulate populations with different initial parameters, the user can directly modify the simulator configuration in :code:`pypopsyn/simulator/configuration.py` or alternatively for a more programmatic way a JSON dictionary containing configuration overrides for the simulation parameters can be provided as a command line argument to this script:

.. code-block:: bash

  python examples/simulator/initialize_evolve_population.py --output_dir test_simulation --parameter_override override.json

This will generate a new folder if it does not exist :code:`test_simulation` in which the simulation results will be dumped: the initial population in compressed binary format `initial_population.pkl.gz`, the final population in the same format `final_population.pkl.gz`, the profiles for the simulation if enabled and the dictionary containing the parameter overrides in `override.json` for reproducibility.

For more information about the simulation script, issue the :code:`--h` argument:

.. code-block:: bash

  python examples/simulator/initialize_evolve_population.py --h

If you want to generate a huge parameter sweep you can use the wrapper or helper script that allows the specification of parameters in a linear spacing format :code:`--parameter [low] [high] [steps]`:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir test_simulator --kick_model "km_exp" --vk_c 100.0 200.0 100

This example will generate a sweep of :code:`100` uniformly spaced samples for the :code:`vk_c` parameter in the range :code:`[100.0, 200.0]` using the :code:`km_exp` kick model and the results will be dumped in the specified :code:`test_simulator` folder.

You can also sweep over more than one parameters by running a script like:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --output_dir test_simulator --kick_model "km_exp" --vk_c 100.0 200.0 100 --h_c 0.01 2 10

In this way a population is simulated for each combination of values of :code:`vk_c` and :code:`h_c`, i.e., the above case corresponds to :code:`100 x 10 = 1000` simulations. The sweeper will generate a folder :code:`test_simulator` which will contain a folder for each simulation (parameter combination).

Here it is important to distinguish between to types of arguments for the helper: options and parameters. Options are arguments which specify procedures for the simulation and they cannot be swept in a range since we assume they are not going to be outputs that will need to be predicted, e.g., the kick model. Parameters are values that we expect to use as ground truth for the learning system and therefore are to be predicted so they can be swept in a range to generate a dataset, e.g., :code:`vk_c`.

The script automatically checks the compatibility of the present parameters for the selected options, e.g., :code:`vk_c` cannot be specified if :code:`km_maxell` has been chosen as kick model. This is done using the dictionary :code:`examples/simulator/config_sweeper.json` which specifies a list of exclusive parameters for each option.
