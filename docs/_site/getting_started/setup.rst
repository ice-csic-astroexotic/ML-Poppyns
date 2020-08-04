*****
Setup
*****

These instructions will provide you with a copy of the project and help you get it up 
and running on your local machine. First, you should clone the repository on your 
computer. The repo contains an environment file that can be installed by running

.. code-block:: bash

  conda env create -f environment.yaml

This environment can be activated using

.. code-block:: bash

  conda activate pop_syn

We recommend working within this environment when using the code. To install the 
`pypopsyn` package and work with the code run

.. code-block:: bash

  python setup.py develop

To automate the workflow and improve as well as maintain code quality standards, we 
have set up pre-commit hooks. To set the hooks run

.. code-block:: bash

  pre-commit install

The steps with pre-commit are as follows: (i) modify code, (ii) stage changes with 
`git add`, (iii) running `git commit` will automatically execute the pre-commit 
framework. If the pre-commit checks are passed, the changes are commit. If not files
are modified and the steps (i) - (iii) have to be repeated. For more info see 
`pre-commit documentation <https://pre-commit.com/#intro>`_ or `this Medium post <https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1>`_.