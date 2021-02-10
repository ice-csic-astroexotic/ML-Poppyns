*******
Logging
*******

For logging information to both console and file outputs we make use of Python's built-in :code:`logging` module.

In each module where the logger needs to be used, you must first import the logging module and get it using the :code:`getLogger` function:

.. code-block:: python

  import logging
  [...]
  log = logging.getLogger(__name__)

Then you can issue logging messages at the appropriate level:

.. code-block:: python

  log.debug("blablabla")
  log.info("blablabla")
  log.warning("blablabla")
  log.error("blablabla")

As an example if we run the script to simulate a population of neutron stars, the following logging information is showed:

.. code-block:: bash

    (pop_syn) michele@michele-XPS-13-7390:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/initialize_evolve_population.py --parameter_override examples/simulator/parameter_override.json  --output_dir simulated_data
    Updating key kick_model in configuration with value km_exp...
    Updating key sigma_k in configuration with value 300...
    INFO:pypopsyn.simulator.initial_population:Seed: 1612514893
    INFO:__main__:Randomizing population age...
    INFO:__main__:Generating initial positions...
    INFO:__main__:Generating initial kick velocities...
    INFO:__main__:Computing orbital velocities...
    INFO:__main__:Computing initial total velocities...
    <prof>[InitialPopulation][Initial] took 2.2616 [s] (cumulative 2.2616 [s])
    <prof>[InitialPopulation][Energy] took 2.5177 [s] (cumulative 4.7793 [s])
    INFO:__main__:Creating data frame for exporting...
    <prof>[InitialPopulation][Export] took 0.3964 [s] (cumulative 5.1757 [s])
    INFO:__main__:Output of the initial population generated in /home/michele/Workspace/MAGNESIA_population_synthesis/initial_population.pkl.gz
    <prof>[InitialPopulation] finished took 5.1763 [s]
    INFO:__main__:Evolving the initial population in time...
    INFO:__main__:Evolving the positions and velocities...
    <prof>[EvolvePopulation][Evolution] took 19.5397 [s] (cumulative 19.5397 [s])
    INFO:__main__:Percentage variation of total energy of the system: -3.7857443555347806e-05 %
    <prof>[EvolvePopulation][Energy] took 0.0057 [s] (cumulative 19.5454 [s])
    INFO:__main__:Creating data frame for exporting...
    INFO:__main__:Output of the evolved population generated in /home/michele/Workspace/MAGNESIA_population_synthesis/final_population.pkl.gz
    <prof>[EvolvePopulation][Export] took 0.5270 [s] (cumulative 20.0724 [s])
    <prof>[EvolvePopulation] finished took 20.0729 [s]

If the :code:`show_profiling` in the :code:`pyposyn/simulator/configuration.py` file is set to :code:`True`, the timing profile for each section of the simulator is also shown on terminal. In general the timing information is saved as a :code:`profile.log` file in the same folder where the output of the simulation is saved.

Until commit :code:`4cb335f7b435f1997cef9c5a9dd84117abff1ff8` Hydra was enabled to allow parameter sweeps when running the simulation script.
By default, Hydra configures the loggers automatically to only produce messages above :code:`info` level, e.g.:

.. code-block:: bash

  (pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/initialize_evolve_population.py
  [2020-03-05 18:34:50,742][__main__][INFO] - Randomizing population age...
  [2020-03-05 18:34:50,743][__main__][INFO] - Generating initial positions...
  [2020-03-05 18:34:57,597][__main__][INFO] - Generating initial proper velocities...
  [2020-03-05 18:34:57,796][__main__][INFO] - Computing orbital velocities...
  [2020-03-05 18:34:58,300][__main__][INFO] - Creating data frame for exporting...
  [2020-03-05 18:34:59,036][__main__][INFO] - Output generated in /home/agarcia/Workspace/MAGNESIA_population_synthesis/outputs/2020-03-05/18-34-50/initial_population.txt

We can configure specific modules to output also debug information by overriding the :code:`hydra.verbose` field providing a list of module names:

.. code-block:: bash

  (pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/initialize_evolve_population.py hydra.verbose=pypopsyn.simulator.initial_population
  [2020-03-05 18:28:47,080][__main__][INFO] - Randomizing population age...
  [2020-03-05 18:28:47,081][pypopsyn.simulator.initial_population][DEBUG] - Drawing random age in range [1.0,100000000.0]
  [2020-03-05 18:28:47,081][__main__][INFO] - Generating initial positions...
  [2020-03-05 18:28:54,013][__main__][INFO] - Generating initial proper velocities...
  [2020-03-05 18:28:54,216][__main__][INFO] - Computing orbital velocities...
  [2020-03-05 18:28:54,723][__main__][INFO] - Creating data frame for exporting...

In this case a log file containing all the experiment's logged messages will be stored in the corresponding folder for the experiments in :code:`outputs` or :code:`multirun` if a multirun sweep is scheduled.