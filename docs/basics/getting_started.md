# Getting started

## Code setup

These instructions will provide you with a copy of the project and help you to get it up and running on your local 
machine. For this you need `conda` to be installed on your machine.
The code has been tested on Ubuntu and macOS.

1. First, you need to clone the repository on your computer. To get the files from our GitHub repository, run
   ```commandline
   git clone https://github.com/csic-ice-magnesia/MAGNESIA_population_synthesis.git
   ```

2. The repo contains an environment file that can be installed by running
   ```commandline
   conda env create -f environment.yaml
   ```

    !!! note

        For macOS users the `cudatoolkit` package has to be commented out in the environment file.
        Otherwise the environment will not be resolved.
   
3. This environment can be activated using
   ```commandline
   conda activate pop_syn
   ```
   To set up the environment on the PIC servers, we specify the full path where the environment will be saved:
   ```commandline
   conda env create --prefix /data/magnesia/scratch/conda/env/pop_syn --file /data/magnesia/software/MAGNESIA_population_synthesis/environment.yaml
   ```
   In this case, the environment can be activated using
   ```commandline
   conda activate /data/magnesia/scratch/conda/env/pop_syn
   ```
   We recommend working within this environment when using the code.

3. To install the `pypopsyn` package locally and work with the code, navigate to the cloned software repository and run
   ```commandline
   python setup.py develop
   ```

4. Finally, to enable full functionality, you need to set the absolute path of the downloaded repository on your local
   machine. To this end, open the configuration file `pypopsyn/simulator/config_simulator.py`, scroll to the section
   titled "GENERAL SIMULATION PARAMETERS" (specifically lines 49 and 50) and add the absolute path to the repository
   folder and to the folder where you would like to save any subsequent simulation output by modifying the variables 
   `cfg["path_to_software"]` and `cfg["path_to_output"]`, respectively.

5. If you also want to use the code to perform machine learning experiments with simulation-based inference, you will
   need to install the [Simulation Based Inference (SBI)](https://sbi-dev.github.io/sbi/>) library after activating the
   environment by running:
   ```commandline
   pip install sbi==0.22.0
   ```

## For developers

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. 
To set the hooks run
```commandline
pre-commit install
```

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with `git add`, (iii) running `git commit`
will automatically execute the pre-commit framework. If the pre-commit checks are passed, the changes are commit. 
If not files are modified and the steps (i) - (iii) have to be repeated. For more info see 
[pre-commit documentation](https://pre-commit.com/#intro>) or [this Medium post](https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1>).


## Documentation

The documentation for this project is held in `docs` and can be compiled into an HTML webpage by running `mkdocs serve`
with the environment activated. The configuration file to set up the documentation with [mkdocs.org](https://www.mkdocs.org) is called `mkdocs.yml` and is located in the main repository folder.


## Repository structure

think about adding this section to the home page

The repository is structured in a modular way to allow for easy adjustments and additions as we continue to improve our software package.
The main folder is `pypopsyn` which contains three sub-folders: `simulator`, `generator`, and `learning`.

* The `simulator` sub-folder contains all the modules and scripts necessary to simulate a population of synthetic neutron stars.
  We group modules according to their physics, i.e., separating those that are associated with the dynamical evolution, the magneto-rotational evolution, the emission in different electromagnetic wavelengths and the modelled surveys.

* The `generator` sub-folder acts as the link between the physics and the machine-learning algorithms.
  It contains all the modules and scripts necessary to represent our mock neutron star population in a way that is suitable for the machine-learning pipeline.
  We, for example, represent the stars' properties as density and feature maps.

* The `learning` sub-folder contains all the modules and scripts necessary for the machine-learning pipeline, including model architectures, initialization techniques, loss function definitions, training schemes and so on.

The `data` folder contains ten main sub-folders: eight sub-folders containing example data created by running different simulator scripts, the generator script and the training and inference scripts, an `observations` sub-folder and a `paper_results` sub-folder.
All the simulation examples provided here have been run by using the default parameters specified in `pypopsyn/simulator/config_simulator.py`.

* The `example_generator_atnf` sub-folder contains an example dataset of feature maps from the observed pulsar population in the [ATNF catalog](https://www.atnf.csiro.au/research/pulsar/psrcat/).

* The `example_generator_magrot` sub-folder contains an example dataset of feature maps from simulated populations.

* The `example_inference_sbi` sub-folder contains the results of an inference run with sbi on some test simulated data.

* The `example_simulation_dyn` sub-folder contains the results of the dynamical evolution of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_dyn.py`.

* The `example_simulation_full_edm` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_edm.py` module to setup the initial conditions.

* The `example_simulation_full_sam` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_sam.py` module to setup the initial conditions.

* The `example_simulation_helper_magrot` sub-folder contains the results of 20 simulations obtained by running the script `utilities/simulation_helper/run_simulation_set.py` and using the `pypopsyn/simulator/simulate_population_magrot_det.py` simulator.

* The `example_simulation_magrot_det` sub-folder contains the results of a simulation of magneto-rotational evolution and detection of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_magrot_det.py`.

* The `example_training_sbi` sub-folder contains the results of a training run with sbi on some test simulated data.

* The `observations` sub-folder contains catalogs with observed data.

* The `paper_results` sub-folder contains data results for our publications.

The folder `docs` contains all the necessary files to produce the documentation.

The `paper_plots` folder contains the notebooks to generate the plots and figures in our papers.

The directory `tutorials` contains two subfolders with jupyter notebooks with examples to run the simulator, generator and learning scripts and analyze the simulations output.

* The `analysis_notebooks` contains some analysis jupyter notebooks that can be used to plot the distributions and features of a simulated population of neutron stars as well as their evolution in time, to check some physical models used in the simulations and to compare a mock population with the real distribution of pulsars.

* The `tutorial_notebooks` sub-folder contains some notebooks to launch simple examples of the simulator, generator and learning scripts.

The directory `utilities` contains some additional python modules and scripts for profiling our code, running simulations in the PIC server, launch several simulations and perform a parameter sweep, sampling distributions or dataframes, perform statistical analysis and plotting settings.

