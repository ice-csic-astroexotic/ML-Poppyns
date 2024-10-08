# Neutron Star Population Synthesis

Neutron star population synthesis code for the ERC project MAGNESIA - The Magnetar Census

The population synthesis framework in this repository integrates population synthesis to model the birth and evolution 
of the population of Galactic neutron stars with deep learning techniques to perform parameter inference and constrain 
their physical properties.

## Repository structure

The repository is structured in a modular way to allow for easy adjustments and additions as we continue to improve our software package.
The main folder is `pypopsyn` which contains three sub-folders: `simulator`, `generator`, and `learning`.

* The `simulator` sub-folder contains all the modules and scripts necessary to simulate a population of synthetic neutron stars.
  We group modules according to their physics, i.e., separating those that are associated with the dynamical evolution, the magneto-rotational evolution, the emission in different electromagnetic wavelengths and the modelled surveys.

* The `generator` sub-folder acts as the link between the physics and the machine-learning algorithms.
  It contains all the modules and scripts necessary to represent our mock neutron star population in a way that is suitable for the machine-learning pipeline.
  We, for example, represent the stars' properties as density and feature maps.

* The `learning` sub-folder contains all the modules and scripts necessary for the machine-learning pipeline, including model architectures, initialization techniques, loss function definitions, training schemes and so on.

The `data` folder contains various sub-folders containing example data created by running different simulator scripts, the generator script and the training and inference scripts, an `observations` sub-folder and a `paper_results` sub-folder.
All the simulation examples provided here have been run by using the default parameters specified in `pypopsyn/simulator/config_simulator.py`.

* The `example_generator_magrot` sub-folder contains an example dataset of feature maps from simulated populations.

* The `example_generator_observed` sub-folder contains an example dataset of feature maps from the observed pulsar population in the [ATNF catalog](https://www.atnf.csiro.au/research/pulsar/psrcat/) and [Meerkat TPA program](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4582P/abstract).

* The `example_inference_nn` sub-folder contains the results of an inference run with a convolutional neural network (CNN) on some test simulated data.

* The `example_inference_sbi` sub-folder contains the results of an inference run with sbi on some test simulated data.

* The `example_learning_nn` sub-folder contains the results of a training run with a CNN on some training simulated data.

* The `example_learning_sbi` sub-folder contains the results of a training run with sbi on some training simulated data.

* The `example_simulation_dyn` sub-folder contains the results of the dynamical evolution of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_dyn.py`.

* The `example_simulation_full_edm` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_edm.py` module to setup the initial conditions.

* The `example_simulation_full_sam` sub-folder contains the results of a full simulation (dynamical + magneto-rotational evolution + detection) of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_full.py` and by using the `pypopsyn/simulator/initial_population_sam.py` module to setup the initial conditions.

* The `example_simulation_helper_magrot` sub-folder contains the results of 20 simulations obtained by running the script `utilities/experiment_helpers/run_simulation_set.py` and using the `pypopsyn/simulator/simulate_population_magrot_det.py` simulator.

* The `example_simulation_magrot_det` sub-folder contains the results of a simulation of magneto-rotational evolution and detection of a population of neutron stars obtained by running the script `pypopsyn/simulator/simulate_population_magrot_det.py`.

* The `observations` sub-folder contains catalogs with observed data.

* The `paper_results` sub-folder contains data results for our publications.

The folder `docs` contains all the necessary files to produce the documentation.

The `paper_plots` folder contains the notebooks to generate the plots and figures in our papers.

The directory `tutorials` contains two subfolders with jupyter notebooks with examples to run the simulator, generator and learning scripts and analyze the simulations output.

* The `analysis_notebooks` contains some analysis jupyter notebooks that can be used to plot the distributions and features of a simulated population of neutron stars as well as their evolution in time, to check some physical models used in the simulations and to compare a mock population with the real distribution of pulsars.

* The `tutorial_notebooks` sub-folder contains some notebooks to launch simple examples of the simulator, generator and learning scripts.

The directory `utilities` contains some additional python modules and scripts for profiling our code, running simulations in the PIC server, launch several simulations and perform a parameter sweep, sampling distributions or dataframes, perform statistical analysis and plotting settings.

