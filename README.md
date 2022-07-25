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
We recommend working within this environment when using the code.
After activating the environment a few Julia packages need to be installed manually by 
running the following command in a terminal
```
julia -e 'using Pkg; Pkg.add.(["PyCall", "OrdinaryDiffEq", "LSODA"])'
```
Finally, to install the `pypopsyn` package and work with the code run
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

## Documentation

Documentation is held in `docs` and can be compiled into an HTML webpage or to a PDF LaTeX file using
`make html` or `make latexpdf` respectively inside the `docs` folder with the environment activated. Both
commands will generate their output in `docs/_build`.

To run code which uses Julia, the call should be made using `python-jl` instead of `python`.