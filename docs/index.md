# Isolated Pulsar Population Synthesis

Neutron star population synthesis code for isolated pulsars developed as part of the ERC project MAGNESIA - 
The Magnetar Census.

!!! note

    The population synthesis framework in this repository integrates population synthesis to model the birth properties 
    and evolution of the population of Galactic neutron stars with deep learning techniques to perform parameter 
    inference and constrain their physical properties.

## Repository structure

!!! tip

    We recommend you read through this page to familiarize yourself with the structure of the code elements before
    moving on to the [Getting started](basics/getting_started.md) page.

### Main code modules

The repository is structured in a modular way to allow for easy adjustments and additions as we continue to improve our software package.
The main folder is `pypopsyn` which contains three subfolders: `simulator`, `generator`, and `learning`.

* The `simulator` subfolder contains all the modules and scripts necessary to simulate a population of synthetic 
  neutron stars. We group modules according to their physics, i.e., separating those that are associated with the 
  dynamical evolution, the magneto-rotational evolution, the emission in different electromagnetic wavelengths and the 
  modelled surveys.

* The `generator` subfolder acts as the link between the physics and the machine-learning algorithms. It contains all 
  the modules and scripts necessary to represent our mock neutron star population in a way that is suitable for the
  machine-learning pipeline. We can, for example, represent the stars' properties as density and feature maps.

* The `learning` subfolder contains all the modules and scripts necessary for the machine-learning pipeline, including
  model architectures, initialization techniques, loss function definitions, training schemes and so on.

### Example data

The `data` folder contains various subfolders with example data created by running different simulator scripts, the 
generator script and the training and inference scripts, an `observations` subfolder and a `paper_results` subfolder.
Each directory contains a `README.md` file with instructions on how to produce a specific dataset.

!!! note

    All the simulation examples provided in the `data` directory were computed using the default parameters 
    specified in the configuration file `pypopsyn/simulator/config_simulator.py`.

* The `example_generator_magrot` subfolder contains an example dataset of feature maps from simulated populations.

* The `example_generator_observed` subfolder contains an example dataset of feature maps from the observed pulsar 
  population in the [ATNF catalog](https://www.atnf.csiro.au/research/pulsar/psrcat/) and [Meerkat TPA program](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4582P/abstract).

* The `example_inference_nn` subfolder contains the results of an inference run with a convolutional neural network
  (CNN) on some test simulated data.

* The `example_inference_sbi` subfolder contains the results of an inference run with sbi on some test simulated data.

* The `example_learning_nn` subfolder contains the results of a training run with a CNN on some training simulated data.

* The `example_learning_sbi` subfolder contains the results of a training run with sbi on some training simulated data.

* The `example_simulation_dyn` subfolder contains the results of the dynamical evolution of a population of neutron 
  stars obtained by running the script `pypopsyn/simulator/simulate_population_dyn.py`.

* The `example_simulation_full_edm` subfolder contains the results of a full simulation (dynamical + magneto-rotational
  evolution + detection) of a population of neutron stars obtained by running the script 
  `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_edm.py` 
  module to set up the initial conditions.

* The `example_simulation_full_sam` subfolder contains the results of a full simulation (dynamical + magneto-rotational
  evolution + detection) of a population of neutron stars obtained by running the script
  `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_sam.py` 
  module to set up the initial conditions.

* The `example_simulation_helper_magrot` subfolder contains the results of 20 simulations obtained by running the 
  script `utilities/experiment_helpers/run_simulation_set.py` and using the `pypopsyn/simulator/simulate_population_magrot_det.py` simulator.

* The `example_simulation_magrot_det` subfolder contains the results of a simulation of magneto-rotational evolution 
  and detection of a population of neutron stars obtained by running the script
  `pypopsyn/simulator/simulate_population_magrot_det.py`.

* The `observations` subfolder contains catalogs with observed data.

* The `paper_results` subfolder contains data results for our publications.

### Documentation files

The folder `docs` contains all the necessary files to produce the documentation.

### Plots from related publications

The `paper_plots` folder contains the notebooks to generate the plots and figures in our papers:

* Analyzing the Galactic Pulsar Distribution with Machine Learning, [Ronchi et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...916..100R/abstract)

* Isolated Pulsar Population Synthesis with Simulation-based Inference, [Graber et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...968...16G/abstract)

### Tutorials

The directory `tutorials` contains two subfolders with jupyter notebooks with examples to run the simulator, generator 
and learning scripts and analyze the simulations output.

* `analysis_notebooks` contains some analysis Jupyter notebooks that can be used to plot the distributions and features 
  of a simulated population of neutron stars as well as its evolution in time. The main purpose of these notebooks is 
  the ability to check the output of the physical models used in the simulations and to compare a mock population with 
  the real distribution of observed pulsars.

* The `tutorial_notebooks` subfolder contains some notebooks to launch simple examples of the simulator, generator 
  and learning scripts. 

!!! tip

    For instructions on how to use the tutorials, see the [simulator tutorial](tutorials/simulator_tutorial.md) and 
    so on, once you have worked through the [Getting started](basics/getting_started.md) page.

### Useful Python scripts

The directory `utilities` contains some additional python modules and scripts for profiling our code, running 
simulations at our computing cluster at PIC, launching several simulations automatically and performing a parameter 
sweep, sampling distributions or dataframes, performing statistical analysis and plotting settings.

