*****************
Simulator Example
*****************

The :code:`examples/simulator/initialize_evolve_population.py` script is responsible for initializing the simulated population from some initial conditions parameters and evolve it in time.
To simulate populations with different initial parameters the package :code:`hydra` is used. One can pass the values of the parameters through a configuration available at :code:`pypopsyn/simulator/configuration.py` or directly on the command prompt. As an example to simulate four population with kick velocities drawn from exponential distributions with four different characteristic kick velocities :code:`vk_c` values one can run the script:

.. code-block:: bash

  python examples/simulator/initialize_evolve_population.py kick_model="km_exp" vk_c=100.,200.,300.,600. -m 


In this way the multirun mode is used and a directory structure :code:`multirun/YYYY-MM-DD/hh-mm-ss/simulated_population_index` is created, where :code:`YYYY-MM-DD` and :code:`hh-mm-ss` folders names are the date and time when the simulation has been performed and the :code:`simulated_population_index` folders names are :code:`0, 1, 2, ...`. In each one of these folders, the output files :code:`initial_population.pkl.gz`, :code:`final_population.pkl.gz`, a log file :code:`initialize_evolve_population.log` and a folder named :code:`.hydra` are created. The file :code:`final_population.pkl.gz` contains all the physical information (like the final position, the final velocity...) of each star after the evolution from the initial configuration saved in :code:`initial_population.pkl.gz`. The folder :code:`.hydra` contains a file named :code:`config.yaml` storing the parameter values from where the population has been initialized.

You can also change the directory where to save the output files. To do that you need to pass an additional argument when running the script above. In paricular if you are not in multirun mode you have to pass the command :code:`hydra.run.dir = "new_directory_path"`, while if you are using the multirun and sweeping parameter values you need to pass the argument :code:`hydra.sweep.dir = "new_directory_path"`. In this way the folders :code:`0, 1, 2, ...` will be saved into the :code:`new_directory_path`.

If you want to generate a huge parameter sweep, you would need to indicate a really large list (each one of the values for the parameter you want to sweep). Hydra is evolving and will accept a way to sweep the parameters in an easy way with a compact notation but for now it only sweeps integers in a linspace fashion :code:`start:end:steps`. To simplify running such large-scale experiments, you can use the wrapper or helper script to make use of such compact notation:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --kick_model "km_exp" --vk_c 100.0 200.0 100


That will generate a sweep of :code:`100` uniformly spaced samples for the :code:`vk_c` parameter in the range :code:`[100.0, 200.0]`.

You can also sweep over more than one parameters by running a script like:

.. code-block:: bash

  python examples/simulator/simulation_helper.py --kick_model "km_exp" --vk_c 100.0 200.0 100 --h_c 0.01 2 10

In this way a population is simulated for each combination of values of :code:`vk_c` and :code:`h_c`, i.e., the above case corresponds to :code:`100 x 10 = 1000` simulations.

Also in this case if you want to change the path where to save the output files, you can pass the additional argument :code:`--output_dir "new_directory_path"`.