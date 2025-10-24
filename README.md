<p align="center">
  <img src="docs/images/full_logo_mlpoppyns.png" width="500">
  <br>
  <em></em>
</p>

## Getting Started

These instructions will provide you with a copy of the project and help you get it up and running on your local machine.
For this, you need `conda` to be installed on your machine. The code has been tested on Ubuntu and macOS.

1. First, you need to clone the repository on your computer. To get the files from our GitHub repository, run
   ```commandline
   git clone https://github.com/ice-csic-astroexotic/ML-Poppyns.git
   ```

2. The repo contains an environment file that can be installed by running
   ```commandline
   conda env create -f environment.yaml
   ```
   NOTE: For macOS users the `cudatoolkit` package has to be commented out in the environment file. Otherwise, the 
   environment will not be resolved.

   On your local machine, this environment can be activated using
   ```commandline
   conda activate mlpoppyns
   ```
   We recommend working within this environment when using the code.

3. To install the `mlpoppyns` package locally and work with the code, navigate to the cloned software repository and run
   ```commandline
   python setup.py develop
   ```
   
4. To install the new conda environment `pop_syn` as an IPython kernel and use it in a Jupyter Notebook, activate
   the environment as shown in step 2 and then run
   ```commandline
   python -m ipykernel install --user --name pop_syn --display-name "pop_syn"
   ```

5. Finally, to enable full functionality, you need to set the absolute path of the downloaded repository on your local 
   machine. To this end, open the configuration file `mlpoppyns/simulator/config_simulator.py`, scroll to the section
   titled "GENERAL SIMULATION PARAMETERS" (specifically lines 49 and 50) and add the absolute path to the repository
   folder and the path to the folder where you would like to save any subsequent simulation output by modifying the 
   variables `cfg["path_to_software"]` and `cfg["path_to_output"]`, respectively. Both paths could, for example, read
   `/home/user/Documents/ML-Poppyns`.

6. If you also want to use the code to perform machine learning experiments with simulation-based inference, you will 
   need to install the [Simulation Based Inference (SBI)](https://sbi-dev.github.io/sbi/) library after activating the 
   environment by running:
   ```commandline
   pip install sbi==0.22.0
   ```
   
## How to use the code?

Navigate to the `tutorials/tutorial_notebook` directory, where several Jupyter Notebooks introduce the different
parts of our code. You can also check out the full documentation for further details (see below).


## Documentation

The documentation for this project is located in the `docs` directory and can be compiled into an HTML webpage by 
running 
```commandline
mkdocs serve
```
with the environment activated. 
This command creates a URL that will be shown in the terminal. Click on it to access the full documentation.


## For Developers

To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. To 
set the hooks run
```commandline
pre-commit install
```
The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with `git add`, (iii) running `git commit` 
will automatically execute the pre-commit framework. If the pre-commit checks are passed, the changes are commit. If not 
files are modified and the steps (i) - (iii) have to be repeated. For more info see [here](https://pre-commit.com/#intro) or [here](https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1).

## Citation

If you use ML-Poppyns in your research, please reference the repository [website](https://github.com/ice-csic-astroexotic/ML-Poppyns) and the following publications: 

* Analyzing the Galactic Pulsar Distribution with Machine Learning, [Ronchi et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...916..100R/abstract)

* Isolated Pulsar Population Synthesis with Simulation-based Inference, [Graber et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...968...16G/abstract)

* Radio pulsar population synthesis with consistent flux measurements using simulation-based inference, [Pardo-Araujo et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...696A.114P/abstract)

## Contacts

If you encounter any issues or have questions, please feel free to email us:

* Vanessa Graber (Vanessa.Graber@rhul.ac.uk)
* Celsa Pardo Araujo (pardo@ices.csic.es)
* Michele Ronchi (ronchi@astron.nl)