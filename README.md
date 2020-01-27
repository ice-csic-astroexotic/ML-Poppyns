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
python examples/generating_population.py
```
creates a file `examples/data/initial_population.py` with the positions for each of the 
simulated neutron stars. The chosen coordinate system is Cartesian with the galactic
centre at its origin. The data file is used in a corresponding jupyter notebook 
`examples/initial_population_plots.ipynb` to plot the simulated neutron star 
distribution.

## Tests

To run all the test available in this repo at once, run 
```
$ python setup.py test
```
which will output a report including coverage.
