# MAGNESIA Population Synthesis
Population synthesis code for the ERC project MAGNESIA - The Magnetar Census

## Getting Started

These instructions will provide you with a copy of the project and help you get it up 
and running on your local machine. First, you should clone the repository on your 
computer. The repo contains an environment file that can be installed by running
```
$ conda env create -f environment.yaml
```
NOTE: for OSX users the `cudatoolkit` package has to be commented out in the environment file.
Furthermore the `julia` package from conda-forge is not available for Mac computers with M1 Apple 
silicon processors. 

This environment can be activated using
```
$ conda activate pop_syn
```
After activating the environment to install the [Simulation Based Inference (SBI)](https://www.mackelab.org/sbi/) 
library run:
```
$ pip install sbi==0.21.0
```
We recommend working within this environment when using the code.

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

## Julia 

To optimise run times, several of our simulation scripts are available in Julia 
(in addition to their native Python versions). The environment created above automatically 
installs Julia. Julia relevant files are located in the folder `julia`. However, after activating the environment a few Julia packages need to be 
installed manually by running the following command in a terminal
```
julia -e 'using Pkg; Pkg.add.(["PyCall", "OrdinaryDiffEq", "LSODA"])'
```
Note that you might have to rebuild the `PyCall` package to link it to the correct Python distribution.
To do so, enter a Julia console by typing `julia` into a terminal. Then type
```
ENV["PYTHON"]=".../anaconda3/envs/pop_syn/bin/python"
```
adjusting the path to the location of your conda environment as needed.
Then, in the Julia console type a `]`, which enters the package manager. Then execute
```
build PyCall
```
which will rebuild the PyCall package with the correct Python distribution.

To run Python code which uses Julia (those files are named `..._julia.py`.), 
the call should be made using `python-jl ...` instead of `python ...`.

You might encounter this error when running Julia code through Python:
```
ImportError: /home/michele/miniconda3/envs/pop_syn/bin/../lib/julia/libstdc++.so.6: 
version `GLIBCXX_3.4.30' not found (required by /home/michele/miniconda3/envs/pop_syn/lib/python3.10/
site-packages/scipy/optimize/_highs/_highs_wrapper.cpython-310-x86_64-linux-gnu.so)
```
If this happens, a possible solution is to run the following command in the same terminal before 
launching the simulation script:
```
export LD_PRELOAD="/home/michele/miniconda3/envs/pop_syn/lib/libstdc++.so.6.0.30"
```
taking care of using your anaconda installation path.

## Documentation

Documentation is held in `docs` and can be compiled into an HTML webpage or to a PDF LaTeX file using
`make html` or `make latexpdf` respectively inside the `docs` folder with the environment activated. Both
commands will generate their output in `docs/_build`.
