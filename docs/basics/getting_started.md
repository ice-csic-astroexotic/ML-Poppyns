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

    !!! warning "macOS users"

        For macOS users the `cudatoolkit` package has to be commented out in the environment file.
        Otherwise the environment will not be resolved.

    To set up the environment on the PIC servers, we specify the full path where the environment will be saved:
    ```commandline
    conda env create --prefix /data/magnesia/scratch/conda/env/pop_syn --file  /data/magnesia/software/MAGNESIA_population_synthesis/environment.yaml
    ```
   
3. On your local machine, the environment can be activated using 
   ```commandline
   conda activate pop_syn
   ```
   On the PIC, the environment can be activated using
   ```commandline
   conda activate /data/magnesia/scratch/conda/env/pop_syn
   ```
   We recommend working within this environment when using the code.

3. To install the `pypopsyn` package locally and use the code, navigate to the cloned software repository and run
   ```commandline
   python setup.py develop
   ```

4. Finally, to enable full functionality, you need to set the absolute path of the downloaded repository on your local
   machine. To this end, open the configuration file `pypopsyn/simulator/config_simulator.py`, scroll to the section
   titled "GENERAL SIMULATION PARAMETERS" (specifically lines 49 and 50) and add the absolute path to the repository
   folder and to the folder where you would like to save any subsequent simulation output by modifying the variables 
   `cfg["path_to_software"]` and `cfg["path_to_output"]`, respectively.

5. If you also want to use the code to perform machine learning experiments with simulation-based inference, you will
   need to install the [Simulation Based Inference (sbi)](https://sbi-dev.github.io/sbi/>) library after activating the
   environment by running:
   ```commandline
   pip install sbi==0.22.0
   ```

!!! tip

    Now that you are ready to work with the code, we recommend taking a look at the different tutorials
    starting with how to simulate a pulsar population (see the 
    [simulator tutorial](../tutorials/simulator_tutorial.md)).

## For developers

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. 
To set the hooks run
```commandline
pre-commit install
```

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with `git add`, (iii) running `git commit`
will automatically execute the pre-commit framework. If the pre-commit checks are passed, the changes are commit. 
If not files are modified and the steps (i) - (iii) have to be repeated. For more info see the 
[pre-commit documentation](https://pre-commit.com/#intro>).


## Documentation

The documentation for this project is held in `docs` and can be compiled into an HTML webpage by running `mkdocs serve`
with the environment activated. The configuration file to set up the documentation with [Materials for MkDocs](https://squidfunk.github.io/mkdocs-material/) is called `mkdocs.yml` and is located in the main repository folder.