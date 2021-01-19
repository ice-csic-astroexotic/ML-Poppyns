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



Logging with Hydra
******************

Until commit :code:`4cb335f7b435f1997cef9c5a9dd84117abff1ff8` Hydra was enabled to allow parameter sweeps when running the simulation script.
By default, Hydra configures the loggers automatically to only produce messages above :code:`info` level, e.g.:

.. code-block:: bash

  (pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/generating_population.py[2020-03-05 18:34:50,742][__main__][INFO] - Randomizing population age...
  [2020-03-05 18:34:50,743][__main__][INFO] - Generating initial positions...
  [2020-03-05 18:34:57,597][__main__][INFO] - Generating initial proper velocities...
  [2020-03-05 18:34:57,796][__main__][INFO] - Computing orbital velocities...
  [2020-03-05 18:34:58,300][__main__][INFO] - Creating data frame for exporting...
  [2020-03-05 18:34:59,036][__main__][INFO] - Output generated in /home/agarcia/Workspace/MAGNESIA_population_synthesis/outputs/2020-03-05/18-34-50/initial_population.txt

We can configure specific modules to output also debug information by overriding the :code:`hydra.verbose` field providing a list of module names:

.. code-block:: bash

  (pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/generating_population.py hydra.verbose=pypopsyn.simulator.initial_population
  [2020-03-05 18:28:47,080][__main__][INFO] - Randomizing population age...
  [2020-03-05 18:28:47,081][pypopsyn.simulator.initial_population][DEBUG] - Drawing random age in range [1.0,100000000.0]
  [2020-03-05 18:28:47,081][__main__][INFO] - Generating initial positions...
  [2020-03-05 18:28:54,013][__main__][INFO] - Generating initial proper velocities...
  [2020-03-05 18:28:54,216][__main__][INFO] - Computing orbital velocities...
  [2020-03-05 18:28:54,723][__main__][INFO] - Creating data frame for exporting...

A log file containing all the experiment's logged messages will be stored in the corresponding folder for the experiments in :code:`outputs` or :code:`multirun` if a multirun sweep is scheduled.