# MAGNESIA Population Synthesis
Population synthesis code for the ERC project MAGNESIA - The Magnetar Census

## Getting Started

These instructions will provide you with a copy of the project and help you get it up and running on your local machine. 
The code has been tested on Ubuntu and macOS.

1. First, you need to clone the repository on your computer. To get the files from our GitHub repository, run
   ```commandline
   git clone https://github.com/csic-ice-magnesia/MAGNESIA_population_synthesis.git
   ```

2. The repo contains an environment file that can be installed by running
   ```commandline
   conda env create -f environment.yaml
   ```
   NOTE: For macOS users the `cudatoolkit` package has to be commented out in the environment file.

   This environment can be activated using
   ```commandline
   conda activate pop_syn
   ```
   We recommend working within this environment when using the code.

3. To install the `pypopsyn` package locally and work with the code, navigate to the cloned software repository and run
   ```commandline
   python setup.py develop
   ```

4. Finally, to enable full functionality, you need to set the absolute path of the downloaded repository on your local 
machine. To this end, open the configuration file `pypopsyn/simulator/config_simulator.py`, scroll to the section
   titled "GENERAL SIMULATION PARAMETERS" (specifically lines 49 and 50) and add the absolute path to the repository
   folder and to the folder where you would like to save any subsequent simulation output by modifying the variables `cfg["path_to_software"]` and `cfg["path_to_output"]`, respectively.

5. If you also want to use the code to perform machine learning experiments with simulation-based inference, you will 
need to install the [Simulation Based Inference (SBI)](https://sbi-dev.github.io/sbi/) library after activating the 
environment by running:
   ```commandline
   pip install sbi==0.22.0
   ```
   
## How to use the code?

Navigate to the `tutorials` directory, where three Jupyter Notebooks introduce the three parts of our code. You can 
also check out our webpage for further details.


## Documentation

Documentation is held in `docs` and can be compiled into an HTML webpage or to a PDF LaTeX file using `make html` or 
`make latexpdf` respectively inside the `docs` folder with the environment activated. Both
commands will generate their output in `docs/_build`.


## For Developers

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. To 
set the hooks run
```commandline
pre-commit install
```
The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with `git add`, (iii) running `git commit` 
will automatically execute the pre-commit framework. If the pre-commit checks are passed, the changes are commit. If not 
files are modified and the steps (i) - (iii) have to be repeated. For more info see [here](https://pre-commit.com/#intro) or [here](https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1).
