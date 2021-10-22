***************
Introduction
***************

Getting started
********************

These instructions will provide you with a copy of the project and help you to get it up
and running on your local machine. First, you should clone the repository on your 
computer. The repo contains an environment file that can be installed by running

.. code-block:: bash

  conda env create -f environment.yaml

This environment can be activated using

.. code-block:: bash

  conda activate pop_syn

We recommend working within this environment when using the code. To install the :code:`pypopsyn` package and work with the code run

.. code-block:: bash

  python setup.py develop

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. To set the hooks run

.. code-block:: bash

  pre-commit install

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with 
:code:`git add`, (iii) running :code:`git commit` will automatically execute the pre-commit
framework. If the pre-commit checks are passed, the changes are commit. If not files
are modified and the steps (i) - (iii) have to be repeated. For more info see 
`pre-commit documentation <https://pre-commit.com/#intro>`_ or `this Medium post <https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1>`_.

The documentation for this project is held in :code:`docs` and can be compiled into an HTML webpage or to a PDF LaTeX file using :code:`make html` or :code:`make latexpdf` respectively inside the :code:`docs` folder with the environment activated. Both commands will generate their output in :code:`docs/_build`.

Repository structure
********************

The repository is structured in a modular way to allow for easy adjustments and additions as we continue to improve our software package.
The main folder is :code:`pypopsyn` which contains four sub-folders: :code:`simulator`, :code:`generator` and :code:`learning` and :code:`benchmark`.

* The :code:`simulator` sub-folder contains all the modules and functions necessary to simulate a population of synthetic neutron stars. We group modules according to their physics, i.e., separating those that are associated with the dynamical evolution, the magneto-rotational evolution, the emission in different electromagnetic wavelengths and the modelled surveys.

* The :code:`generator` sub-folder acts as the link between the physics and the machine-learning algorithms. It contains all the modules and functions necessary to represent our mock neutron star population in a way that is suitable for the machine-learning pipeline. We, for example, represent the stars' properties as density and feature maps.

* The :code:`learning` sub-folder contains all the modules and functions necessary for the machine-learning pipeline, including model architectures, initialization techniques, loss function definitions, training schemes and so on.

* The :code:`benchmark` sub-folder contains all the modules and functions necessary to profile our code. We use this functionality to optimize the run-time of our code.


The :code:`examples` folder contains four main sub-folders: :code:`simulator`, :code:`generator`, :code:`learning` and :code:`data`.
The purpose of these examples is to demonstrate the functionality and usage of the respective modules and functions in :code:`pypopsyn`.

* The :code:`simulator` sub-folder contains various scripts that simulate a population of neutron stars (by default the one specified in :code:`pypopsyn/simulator/configuration.py`) from its dynamical evolution to the detection with different surveys as well as a :code:`simulation_helper.py` script to execute a range of simulations in an automated way.

* The :code:`generator` sub-folder contains the script :code:`generate_dataset.py` that reads the simulated data and creates a dataset of feature maps.

* The :code:`learning` sub-folder contains the script :code:`train.py` that trains a neural network on the provided dataset and the script :code:`infer.py` that tests the predictive power of a trained neural network on a test dataset.

* The :code:`data` sub-folder contains examples of simulated data, generated dataset and inference results.

There are also other folders containing additional materials.
For example the directory :code:`notebooks` contains several jupyter notebooks that can be used to plot the distributions and features of a simulated population of neutron stars as well as their evolution in time, the statistics of the inference results of a trained neural network and finally a simple comparison of our mock population with the real distribution of pulsars.
The directory :code:`scripts` contains a couple of additional python scripts, including :code:`pop_sampler.py`, which we use to down-sample our large mock population according to some weight function.
The directory :code:`utilities` contains some modules for statistical analysis and plotting settings.


