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

As an example if we run the script to simulate a population of 100 neutron stars using the standard configuration file, the following logging information is showed:

.. code-block:: bash

    (pop_syn) michele@michele-XPS-13-7390:~/Documents/Magnesia-PhD/MAGNESIA_population_synthesis$ python pypopsyn/simulator/simulate_population_full.py
    INFO:__main__:Seed: 1716554090
    INFO:__main__:Randomizing population age...
    INFO:__main__:Generating initial positions...
    INFO:__main__:Generating initial kick velocities...
    INFO:__main__:Computing orbital velocities...
    INFO:__main__:Computing initial total velocities...
    <prof>[InitialPopulation][Initial position and velocity] took 0.0670 [s] (cumulative 0.0670 [s])
    <prof>[InitialPopulation][Initial energy] took 1.6573 [s] (cumulative 1.7243 [s])
    <prof>[InitialPopulation][Initial angular momentum] took 0.2674 [s] (cumulative 1.9917 [s])
    INFO:__main__:Computing initial field strengths...
    INFO:__main__:Computing initial misalignment angles...
    INFO:__main__:Computing initial periods...
    <prof>[InitialPopulation][Initial field strengths, misalignment angles and periods] took 0.0010 [s] (cumulative 1.9926 [s])
    INFO:__main__:Computing initial period derivatives...
    <prof>[InitialPopulation][Initial period derivatives] took 0.0003 [s] (cumulative 1.9930 [s])
    INFO:__main__:Creating data frame for exporting...
    <prof>[InitialPopulation][Export] took 0.0019 [s] (cumulative 1.9948 [s])
    INFO:__main__:Output of the initial population generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/initial_population.pkl.gz
    <prof>[InitialPopulation] finished took 1.9952 [s]
    INFO:__main__:Evolve the initial population in time dynamically...
    INFO:__main__:Evolving the positions and velocities...
    <prof>[DynamicalEvolution][Dynamical evolution] took 0.0441 [s] (cumulative 0.0441 [s])
    INFO:__main__:Percentage variation of total energy of the system: 4.1472843296823104e-05 %
    <prof>[DynamicalEvolution][Final energy] took 0.0004 [s] (cumulative 0.0445 [s])
    INFO:__main__:Percentage variation of z-component of total angular momentum of the system: 8.043814409866621e-06 %
    <prof>[DynamicalEvolution][Final angular momentum] took 0.0003 [s] (cumulative 0.0448 [s])
    <prof>[DynamicalEvolution] finished took 0.0450 [s]
    INFO:__main__:Evolving magnetic field, misalignment angle and rotation period...
    <prof>[MagnetoRotationalEvolution][Final field strengths, misalignment angles and periods] took 0.0597 [s] (cumulative 0.0597 [s])
    INFO:__main__:Computing final period derivatives...
    <prof>[MagnetoRotationalEvolution][Final period derivatives] took 0.0006 [s] (cumulative 0.0603 [s])
    <prof>[MagnetoRotationalEvolution] finished took 0.0605 [s]
    INFO:__main__:Fraction of pulsars beaming towards us in radio: 0.12
    <prof>[RadioEmission][Radio emission] took 0.0006 [s] (cumulative 0.0006 [s])
    <prof>[RadioEmission] finished took 0.0009 [s]
    INFO:__main__:Fraction of pulsars in the covered sky region: 0.83
    <prof>[RadioDetection][Total sky coverage] took 0.0004 [s] (cumulative 0.0004 [s])
    <prof>[RadioDetection][DM computation] took 0.0106 [s] (cumulative 0.0110 [s])
    INFO:__main__:Simulate detection with PMPS...
    INFO:__main__:Fraction of detected pulsars by PMPS: 0.0
    INFO:__main__:Simulate detection with SMPS...
    INFO:__main__:Fraction of detected pulsars by SMPS: 0.0
    INFO:__main__:Simulate detection with HTRU low latitude...
    INFO:__main__:Fraction of detected pulsars by HTRU low latitude: 0.0
    INFO:__main__:Simulate detection with HTRU mid latitude...
    INFO:__main__:Fraction of detected pulsars by HTRU mid latitude: 0.0
    INFO:__main__:Simulate detection with HTRU high latitude...
    INFO:__main__:Fraction of detected pulsars by HTRU high latitude: 0.0
    <prof>[RadioDetection][Radio surveys detection] took 0.1605 [s] (cumulative 0.1715 [s])
    <prof>[RadioDetection] finished took 0.1720 [s]
    INFO:__main__:Creating data frame for exporting...
    INFO:__main__:Output of the evolved population generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/final_population.pkl.gz
    INFO:__main__:Output of the PMPS survey generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/survey_PMPS_results.pkl.gz
    INFO:__main__:Output of the SMPS survey generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/survey_SMPS_results.pkl.gz
    INFO:__main__:Output of the HTRU high-latitude survey generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/survey_HTRU_high_results.pkl.gz
    INFO:__main__:Output of the HTRU low and mid-latitude surveys generated in /home/michele/Documents/Magnesia-PhD/MAGNESIA_population_synthesis/toremove/sim_full_test/survey_HTRU_low_mid_results.pkl.gz
    <prof>[RadioDetection][Export] took 0.0081 [s] (cumulative 0.1801 [s])
    <prof>[TotalSimulation] finished took 2.2832 [s]



If the :code:`show_profiling` in the :code:`pyposyn/simulator/config_simulator.py` file is set to :code:`True`, the timing profile for each section of the simulator is also shown on terminal.
In general the timing information is saved as a :code:`profile.log` file in the same folder where the output of the simulation is saved.
Function-specific profiling can also be activated by setting :code:`enable_profiles` and / or :code:`show_profiles` to :code:`True`.
The detailed information will be saved in the folder specified under :code:`profiles_dir`.

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