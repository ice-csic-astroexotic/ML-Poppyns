****************
Hydra (outdated)
****************

The current version of the code uses the multi-thread functionality to run multiple processes and allow parameter sweeps and Hydra is no longer available.
To restore the code to a version where Hydra was enabled you should checkout the commit :code:`4cb335f7b435f1997cef9c5a9dd84117abff1ff8`. We used `Hydra <https://hydra.cc/docs/intro>`_ for experiment running and `OmegaConf <https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation>`_ for configuration files and dictionaries. Hydra
is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

Instead of hardcoding the simulator parameters, we decided to outsorce them to a configuration dictionary :code:`simulator/configuration.py`. Such configuration can be overridden via Hydra by providing a YAML configuration file or :term:`CLI` arguments.

For instance, we could run the previous initial population example like this:

::

  python examples/simulator/initialize_evolve_population.py r_extent=30.0

to override the value of the :code:`r_extent` to be :math:`30.0` instead of the default configuration value of :math:`20.0` in :code:`configuration.py`.

Hydra will take care of creating a folder for such experiment by creating an :code:`outputs` folder in the directory where you ran the script. One folder for each experiment (by default with the current datetime) will be created and all the information needed to replicate the experiment plus its output will be stored there.

Another cool feature of Hydra is multi-run. Suppose you want to explore a range of parameters, you could do:

::

  python examples/simulator/generating_population.py r_extent=30.0,40.0 -m

Hydra will automatically take care of executing the script sweeping all the parameters and creating a job for each one of them :math:`30.0` and :math:`40.0`. That will produce an output like:

::

  [2020-03-04 15:54:07,360][HYDRA] Sweep output dir : multirun/2020-03-04/15-54-07
  [2020-03-04 15:54:07,360][HYDRA] Launching 2 jobs locally
  [2020-03-04 15:54:07,360][HYDRA]        #0 : r_extent=30.0
  Updating key r_extent in configuration with value 30.0...
  /home/agarcia/Workspace/magnesia/MAGNESIA_population_synthesis/multirun/2020-03-04/15-54-07/0
  [2020-03-04 15:54:14,620][HYDRA]        #1 : r_extent=40.0
  Updating key r_extent in configuration with value 40.0...
  /home/agarcia/Workspace/magnesia/MAGNESIA_population_synthesis/multirun/2020-03-04/15-54-07/1