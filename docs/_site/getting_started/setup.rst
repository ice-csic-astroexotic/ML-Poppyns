***************
Introduction
***************

Getting started
********************

These instructions will provide you with a copy of the project and help you to get it up and running on your local machine.
For this you need conda to be installed on your machine.
First, you should clone the repository on your computer.
The repo contains an environment file that can be installed by running

::

  conda env create -f environment.yaml

NOTE: for OSX users the :code:`cudatoolkit` package has to be commented out in the environment file.

This environment can be activated using

::

  conda activate pop_syn

To set up the environment on the PIC servers, we specify the full path where the environment will be saved:

::

  conda env create --prefix /data/magnesia/scratch/conda/env/pop_syn
  --file /data/magnesia/software/MAGNESIA_population_synthesis/environment.yaml

In this case, the environment can be activated using

::

  conda activate /data/magnesia/scratch/conda/env/pop_syn

After activating the environment to install the `Simulation Based Inference (SBI) <https://www.mackelab.org/sbi/>`_
library run:

::

   pip install sbi

We recommend working within this environment when using the code. To install the :code:`pypopsyn` package and work with the code run

::

  python setup.py develop

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. To set the hooks run

::

  pre-commit install

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with 
:code:`git add`, (iii) running :code:`git commit` will automatically execute the pre-commit
framework. If the pre-commit checks are passed, the changes are commit. If not files
are modified and the steps (i) - (iii) have to be repeated. For more info see 
`pre-commit documentation <https://pre-commit.com/#intro>`_ or `this Medium post <https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1>`_.

Finally you should setup the absolute path to where the repository is saved on your local machine.
This can be done by opening the configuration file :code:`pypopsyn/simulator/config_simulator.py` and in the section named
"GENERAL SIMULATION PARAMETERS" adding the absolute path to the repository folder by modifying the variable
:code:`cfg["path_to_software"]` under the :code:`else` statement.


Documentation
*************

The documentation for this project is held in :code:`docs` and can be compiled into an HTML webpage or to a PDF
LaTeX file using :code:`make html` or :code:`make latexpdf` respectively inside the :code:`docs` folder with the
environment activated. Both commands will generate their output in :code:`docs/_build`.

Repository structure
********************

The repository is structured in a modular way to allow for easy adjustments and additions as we continue to improve our software package.
The main folder is :code:`pypopsyn` which contains five sub-folders: :code:`simulator`, :code:`generator` and :code:`learning` and :code:`benchmark`.

* The :code:`simulator` sub-folder contains all the modules and scripts necessary to simulate a population of synthetic neutron stars.
  We group modules according to their physics, i.e., separating those that are associated with the dynamical evolution, the magneto-rotational evolution, the emission in different electromagnetic wavelengths and the modelled surveys.

* The :code:`generator` sub-folder acts as the link between the physics and the machine-learning algorithms.
  It contains all the modules and scripts necessary to represent our mock neutron star population in a way that is suitable for the machine-learning pipeline.
  We, for example, represent the stars' properties as density and feature maps.

* The :code:`learning` sub-folder contains all the modules and scripts necessary for the machine-learning pipeline, including model architectures, initialization techniques, loss function definitions, training schemes and so on.

The :code:`data` folder contains seven main sub-folders: five sub-folders containing example data created by running different simulator scripts and the generator script, an :code:`observations` sub-folder and a :code:`paper_results` sub-folder.
All the simulation examples provided here have been run by using the default parameters specified in :code:`pypopsyn/simulator/config_simulator.py`.

* The :code:`example_simulation_dyn` sub-folder contains the results of the dynamical evolution of a population of neutron stars obtained by running the script :code:`pypopsyn/simulator/simulate_population_dyn.py`.

* The :code:`example_simulation_full_sam` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script :code:`pypopsyn/simulator/simulate_population_full.py` and by using the :code:`pypopsyn/simulator/initial_population_sam.py` module to setup the initial conditions.

* The :code:`example_simulation_full_edm` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script :code:`pypopsyn/simulator/simulate_population_full.py` and by using the :code:`pypopsyn/simulator/initial_population_edm.py` module to setup the initial conditions.

* The :code:`example_simulation_magrot_det` sub-folder contains the results of a simulation of magneto-rotational evolution and detection of a population of neutron stars obtained by running the script :code:`pypopsyn/simulator/simulate_population_magrot_det.py`.

* The :code:`example_generator` sub-folder contains an example dataset of feature maps from simulated populations.

* The :code:`observations` sub-folder contains catalogs with observed data.

* The :code:`paper_results` sub-folder contains data results for our publications.

There are also other folders containing additional materials.
For example the directory :code:`tutorials` contains two subfolders with jupyter notebooks with examples to run the simulator, generator and learning scripts and analyze the simulations output.

* The :code:`getting_started` sub-folder contains some notebooks to launch simple examples of the simulator, generator and learning scripts.

* The :code:`analysis_notebooks` contains some analysis jupyter notebooks that can be used to plot the distributions and features of a simulated population of neutron stars as well as their evolution in time, to check some physical models used in the simulations and to compare a mock population with the real distribution of pulsars.

The directory :code:`utilities` contains some additional python modules and scripts for profiling our code, running simulations in the PIC server, launch several simulations and perform a parameter sweep, sampling distributions or dataframes, perform statistical analysis and plotting settings.


