***************
Getting started
***************

These instructions will provide you with a copy of the project and help you get it up 
and running on your local machine. First, you should clone the repository on your 
computer. The repo contains an environment file that can be installed by running

.. code-block:: bash

  conda env create -f environment.yaml

This environment can be activated using

.. code-block:: bash

  conda activate pop_syn

We recommend working within this environment when using the code. To install the 
`pypopsyn` package and work with the code run

.. code-block:: bash

  python setup.py develop

To automate the workflow and improve as well as maintain code quality standards, we 
have set up pre-commit hooks. To set the hooks run

.. code-block:: bash

  pre-commit install

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with 
`git add`, (iii) running `git commit` will automatically execute the pre-commit 
framework. If the pre-commit checks are passed, the changes are commit. If not files
are modified and the steps (i) - (iii) have to be repeated. For more info see 
`pre-commit documentation <https://pre-commit.com/#intro>`_ or `this Medium post <https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1>`_.

Repository structure
********************

The repository is structured in a modular way.
The main folder is :code:`pypopsyn` which contains four sub-folders: :code:`simulator`, :code:`generator` and :code:`learning` and :code:`benchmark`.

* The :code:`simulator` sub-folder contains all the modules and functions necessary to simulate a population of neutron stars.

* The :code:`generator` sub-folder contains all the modules and functions necessary to create a dataset of density and velocity maps.

* The :code:`learning` sub-folder contains all the modules and functions necessary for the machine learning pipeline.

* The :code:`benchmark` sub-folder contains all the modules and functions necessary for the time profiling of the whole code.


The :code:`examples` folder contains four main sub-folders: :code:`simulator`, :code:`generator`, :code:`learning` and :code:`data`.
The purpose of these examples is to demonstrate the functionality and usage of the respective modules and functions in :code:`pypopsyn`.

* The :code:`simulator` sub-folder contains the script :code:`initialize_evolve_population.py` that simulate the population of neutron stars.

* The :code:`generator` sub-folder contains the script :code:`generate_dataset.py` that read the simulated data and create a dataset of density and velocity maps.

* The :code:`learning` sub-folder contains the script :code:`train.py` that trains a neural network on the provided dataset and the script :code:`infer.py` that tests the predictive power of a trained neural network on a test dataset.

* The :code:`data` sub-folder contains examples of simulated data, generated dataset and inference results.

There are also other folders containing additional material.
For example the directory :code:`notebooks` contains some jupyter notebooks that can be used to plot the distributions and features of simulated population of neutron stars, and the statistic of the inference results of a trained neural network.
The directory :code:`scripts` contains a couple of additional python scripts.
The directory :code:`utilities` contains some modules for statistical analysis and plotting settings.


