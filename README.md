# MAGNESIA Population Synthesis
Population synthesis code for the ERC project MAGNESIA - The Magnetar Census

## Getting Started

These instructions will provide you with a copy of the project and help you get it up and running on your local machine.
First, you should clone the repository on your computer. The repo contains an environment file that can be installed by running
```
$ conda env create -f environment.yaml
```
This environment can be activated using
```
$ conda activate pop_syn
```
We recommend working within this environment when using the code. To install the package and work with the code run
```
$ python setup.py develop
```
To automate the workflow and improve as well as maintain code quality standards, we have set up pre-commit hooks. To set the hooks run
```
$ pre-commit install
```

## Tests

To run all the test available in this repo at once, run 
```
$ python setup.py test
```
which will output a report including coverage.
