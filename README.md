# MAGNESIA Population Synthesis
Population synthesis code for the ERC project MAGNESIA - The Magnetar Census

## Getting Started

These instructions will provide you with a copy of the project and help you get it up 
and running on your local machine. First, you should clone the repository on your 
computer. The repo contains an environment file that can be installed by running
```
$ conda env create -f environment.yaml
```
This environment can be activated using
```
$ conda activate pop_syn
```
We recommend working within this environment when using the code. To install the 
`pypopsyn` package and work with the code run
```
$ python setup.py develop
```
To automate the workflow and improve as well as maintain code quality standards, we 
have set up pre-commit hooks. To set the hooks run
```
$ pre-commit install
```
The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with 
`git add`, (iii) running `git commit` will automatically execute the pre-commit 
framework. If the pre-commit checks are passed, the changes are commit. If not files
are modified and the steps (i) - (iii) have to be repeated. For more info see 
[here](https://pre-commit.com/#intro) or [here](https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1).

## Code Structure

## Example

A file `generating_population.py` that simulates an initial 
population of neutron stars is provided in the `examples` folder. Running 
```
python examples/simulator/generating_population.py
```
creates a file `initial_population.txt` with the positions for each of the 
simulated neutron stars. The chosen coordinate system is Cartesian with the galactic
centre at its origin. The data file can be used in a corresponding jupyter notebook 
`examples/simulator/initial_population_plots.ipynb` to plot the simulated neutron star 
distribution.

## Experiment Configuration and Hydra

We are using [Hydra](https://hydra.cc/docs/intro) for experiment running and [OmegaConf](https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation) for configuration files and dictionaries and. Hydra 
is an open-source Python framework that simplifies the development of research and
other complex applications. The key feature is the ability to dynamically create a
hierarchical configuration by composition and override it through config files and
the command line. The name Hydra comes from its ability to run multiple similar
jobs - much like a Hydra with multiple heads.

Instead of hardcoding the simulator parameters, we decided to outsorce them to
a configuration dictionary `simulator/configuration.py`. Such configuration can
be overridden via Hydra by providing a `YAML` configuration file or CLI arguments.

For instance, we could run the previous initial population example like this:

```
python examples/simulator/generating_population.py r_extent=30.0
```

to override the value of the `r_extent` to be `30.0` instead of the default
configuration value of `20.0` in `configuration.py`.

Hydra will take care of creating a folder for such experiment by creating an
`outputs` folder in the directory where you ran the script. One folder for
each experiment (by default with the current datetime) will be created and
all the information needed to replicate the experiment plus its output will
be stored there.

Another cool feature of Hydra is multi-run. Suppose you want to explore a range
of parameters, you could do:

```
python examples/simulator/generating_population.py r_extent=30.0,40.0 -m
```

Hydra will automatically take care of executing the script sweeping all the
parameters and creating a job for each one of them `30.0` and `40.0`. That will
produce an output like:

```
[2020-03-04 15:54:07,360][HYDRA] Sweep output dir : multirun/2020-03-04/15-54-07
[2020-03-04 15:54:07,360][HYDRA] Launching 2 jobs locally
[2020-03-04 15:54:07,360][HYDRA]        #0 : r_extent=30.0
Updating key r_extent in configuration with value 30.0...
/home/agarcia/Workspace/magnesia/MAGNESIA_population_synthesis/multirun/2020-03-04/15-54-07/0
[2020-03-04 15:54:14,620][HYDRA]        #1 : r_extent=40.0
Updating key r_extent in configuration with value 40.0...
/home/agarcia/Workspace/magnesia/MAGNESIA_population_synthesis/multirun/2020-03-04/15-54-07/1
```

## Logging

For logging informatio to both console and file outputs we make use of Python's
built-in `logging` module in combination with Hydra.

In each module where the logger needs to be used, you must first import the
logging module and get it using the `getLogger` function:

```
import logging
[...]
log = logging.getLogger(__name__)
```

Then you can issue logging messages at the appropriate level:

```
log.debug("blablabla")
log.info("blablabla")
log.warning("blablabla")
log.error("blablabla")
```

By default, Hydra configures the loggers automatically to only produce messages
above `info` level, e.g.:

```
(pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/generating_population.py[2020-03-05 18:34:50,742][__main__][INFO] - Randomizing population age...
[2020-03-05 18:34:50,743][__main__][INFO] - Generating initial positions...
[2020-03-05 18:34:57,597][__main__][INFO] - Generating initial proper velocities...
[2020-03-05 18:34:57,796][__main__][INFO] - Computing orbital velocities...
[2020-03-05 18:34:58,300][__main__][INFO] - Creating data frame for exporting...
[2020-03-05 18:34:59,036][__main__][INFO] - Output generated in /home/agarcia/Workspace/MAGNESIA_population_synthesis/outputs/2020-03-05/18-34-50/initial_population.txt
```

We can configure specific modules to output also debug information by overriding
the `hydra.verbose` field providing a list of module names:

```
(pop_syn) agarcia@challenger:~/Workspace/MAGNESIA_population_synthesis$ python examples/simulator/generating_population.py hydra.verbose=pypopsyn.simulator.initial_population
[2020-03-05 18:28:47,080][__main__][INFO] - Randomizing population age...
[2020-03-05 18:28:47,081][pypopsyn.simulator.initial_population][DEBUG] - Drawing random age in range [1.0,100000000.0]
[2020-03-05 18:28:47,081][__main__][INFO] - Generating initial positions...
[2020-03-05 18:28:54,013][__main__][INFO] - Generating initial proper velocities...
[2020-03-05 18:28:54,216][__main__][INFO] - Computing orbital velocities...
[2020-03-05 18:28:54,723][__main__][INFO] - Creating data frame for exporting...
[2020-03-05 18:28:55,462][__main__][INFO] - Output generated in /home/agarcia/Workspace/MAGNESIA_population_synthesis/outputs/2020-03-05/18-28-47/initial_population.txt
```

A log file containing all the experiment's logged messages will be stored in the
corresponding folder for the experiments in `outputs` or `multirun` if a multirun
sweep is scheduled.

## Tests

To run all the test available in this repo at once, run 
```
$ python setup.py test
```
which will output a report including coverage.
