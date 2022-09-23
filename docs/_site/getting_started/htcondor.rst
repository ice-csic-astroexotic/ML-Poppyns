################################################
HTCondor
################################################

This page is primarily for MAGNESIA developers. We focus on explaining the file structure and main commands needed to submit simulations using the HTCondor infrastructure at PIC. The documentation provided by PIC about HTCondor can be found at https://pwiki.pic.es/index.php?title=HTCondor_User_Guide

**********************
Location of the files
**********************

#. The main software repo is located in the folder :code:`/data/magnesia/software`. Note that the repo was cloned in this location following PIC's guidelines: we should use the software folder for shared repositories. The repo was cloned in such a way that every MAGNESIA user has writing, execution and reading access. A user would, in principle, be able to clone the repo in their own home directories, but this should be avoided due to limited disk space. Before lunching simulations from any location in the server, the variable :code:`server_run` in the :code:`configuration.py` file should be set to :code:`true` to ensure that the correct path to the software modules is set.
#. The scripts to submit a job with HTCondor are in the common folder :code:`/data/magnesia/common`. We use this folder to store the HTCondor files since it is PIC's recommended location for storing intermediate data. This is also where we store our simulations and ML experiments on intermediate timescales (before moving them to long-term storage).
#. There is also a scratch folder in MAGNESIA's disk space where output files of each run could be saved. These will however be deleted after each run and would thus need to be transferred elsewhere if required. Below we explain how to transfer files from this scratch directory.

***************************
Useful commands in HTCondor
***************************

The main commands when using HTCondor are the following:

#. :code:`condor_submit file.submit`  Submit the job(s) specified inside :code:`file.submit`.
#. :code:`condor_q`  Query the status of the submitted jobs.
#. :code:`condor_rm <id_job>`   Remove a job.
#. :code:`condor_q -const 'JobStatus == 5' -af HoldReason`  Output the reason why jobs are being held.
#. :code:`condor_ssh_to_job <id_job>`  Enter the working node where the job is running.
#. :code:`condor_submit -i test.sub`  Submit job(s) in interactive mode.

**************************************
Steps to run the dynamical simulations
**************************************

SSH sessions
************

Using the JupyterHub online interface at https://jupyter.pic.es/ we are able to run and read files in the :code:`software` folder. However, to have writing access to this folder we need to log in via ssh. To do so, type the following command in a terminal:

.. code-block:: bash

    ssh user@ui.pic.es

where :code:`user` is your PIC user name. You will be prompted to enter your password.
Currently access via UAB's wireless and ethernet network does not allow ssh connections; in particular PIC uses port 22 which is blocked. As a result a standard ssh connection from the ICE cannot be established at the moment.  However, we can establish a ssh connection through a terminal session within https://jupyter.pic.es/. A standard ssh connection from outside the ICE can however be readily established.

The software repository located in :code:`/data/magnesia/software` should be used to execute large experiments. Any changes, developments or updates of the code itself should be done on personal laptops whenever possible. To update the repository on the PIC servers, we use :code:`git pull`. To establish GitHub access to the repository follow these steps:

  #. Paste the text below (in your ssh session), substituting your GitHub email address  :code:`ssh-keygen -t ed25519 -C "your_email@example.com"` This creates a new SSH key, using the provided email as a label.
  #. When you are prompted to "Enter a file in which to save the key," enter :code:`/data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  #. Start the ssh-agent in the background by entering :code:`eval "$(ssh-agent -s)"`.
  #. Add your SSH private key to the ssh-agent by entering :code:`ssh-add /data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  #. Then follow these instructions to add your new ssh-key to GitHub: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

.. note::
    To :code:`git pull` and :code:`git push` at a later stage, steps 3 and 4 have to be executed every time using a ssh session.

Submit files
************

In  order to submit jobs with HTCondor we need two different scripts: an HTCondor submit file and a wrapper (see below). The former is the file needed to submit a job with HTCondor, while the latter takes care of executing python relevant commands as well as running our python scripts. Relevant example scripts used in the following are saved in :code:`/data/magnesia/common/test_htcondor`.

The HTCondor submit file looks like the following:

.. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ cat test.submit
    # The UNIVERSE defines an execution environment. You will always use VANILLA.

    universe        = vanilla

    # Executable is the program your job will run.
    executable      = wrapper.sh

    # Location of error and output channels from your job
    # that HTCondor returns from the remote host.
    output          = OUTPUT/hello.out.$(Cluster).$(Process).txt
    error           = OUTPUT/hello.error.$(Cluster).$(Process).txt
    log             = OUTPUT/hello.log.$(Cluster).$(Process).txt
    queue

In our case, the executable is a wrapper (explained below) where we call the .py file. When submitting an HTCondor job, simulations are run on a remote host. To see the terminal output (stdout) or errors (stderr) arising during the execution, we save the details in the path specified in the output, log and error variables. The path :code:`OUTPUT/hello.out.$(Cluster).$(Process).txt` is an example. You can choose the path that is most convenient for your purpose. In our example, we want the output to be saved in a folder called :code:`OUTPUT`, in the same location as the submit file and with the name :code:`hello.out.$(Cluster).$(Process).txt`. For example, we could change the path of the output to

.. code-block:: bash

    output = test.txt

and the output (whatever is printed in the terminal during the execution of the wrapper) will be saved in the same folder as the submit file with the name :code:`test.txt`. Note that if you want the path to be as in the previous example (:code:`OUTPUT/hello.out.$(Cluster).$(Process).txt`), you have to first (manually) create the :code:`OUTPUT` directory in the location of the submit file; if the directory is not created first an error will arise. In the script above, :code:`$(Cluster)` represents the cluster identifier and :code:`$(Process)` the process identifier. We use the cluster id and process id to differentiate between different runs and jobs. :code:`Queue` is the start "button".

.. note::
    Files saved in the scratch directory are removed after the execution. To transfer these files to a directory, we need to specify this in the submit file.

For example, if the output file is called :code:`output1.txt` and we want to keep that file, we need to add the following line to the submit file:

.. code-block:: bash

    transfer_output_files= output1.txt

Then :code:`output1.txt` will be saved in the same directory as the submit file.

Wrappers
********

An example wrapper looks like this:

.. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ cat wrapper.sh
    #!/bin/bash

    # Set where the anaconda installation is located in order to be able to use conda commands.
    export PATH=/data/magnesia/software/anaconda3/bin:$PATH

    # Initialize anaconda in the bash shell.
    conda init bash

    # It's recommended by anaconda to close and restart the terminal after conda init.
    source /data/magnesia/software/anaconda3/etc/profile.d/conda.sh

    # Activate conda environment.
    conda activate /data/magnesia/software/anaconda3/envs/pop_syn

    # Run the simulation specifying the path for the output.
    python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn_test.py --output /data/magnesia/common/test_HTCondor

.. note::
    We cannot save the output from the simulations in the same folder as the repo since the remote host does not have writing access in :code:`software`. Thus, we save the output in the :code:`common` folder.

Submitting jobs and related queries
***********************************

In order to submit a job (= running the simulation on the server), we use

.. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ condor_submit test.submit

and it should return

.. code-block:: bash

    Submitting job(s).
    1 job(s) submitted to cluster 5889056.

To look at the status of the jobs that we have submitted, run

.. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ condor_q

with the expected output

.. code-block:: bash

    -- Schedd: submit01.pic.es : <193.109.174.82:9618?... @ 02/18/22 18:39:10
    OWNER    BATCH_NAME     SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
    cpardoar ID: 5888746   2/18 10:32      _      1      _      1 5888746.0
    cpardoar ID: 5888825   2/18 16:13      _      1      _      1 5888825.0

    Total for query: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
    Total for cpardoar: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
    Total for all users: 1081 jobs; 0 completed, 0 removed, 256 idle, 818 running, 7 held, 0 suspended

In this example, we have 2 jobs running. Note that if you have initiated a session at https://jupyter.pic.es/, it will appear as a running job. If we see that some jobs are IDLE, these are currently in HTCondor's queue and waiting to be launched. If you see a job that is on HOLD, this might indicate that something went wrong. To see what happened to held jobs run the following command:

.. code-block:: bash

    condor_q -const 'JobStatus == 5' -af HoldReason

If you want to remove a job, first execute :code:`condor_q` to search for the :code:`job_id` of the job you want to remove. For example, with the output from :code:`condor_q` above, removing the job with the id 5888825.0 can be achieved by running:

.. code-block:: bash

    condor_rm 5888825.0

.. note::
    The first job shown by :code:`condor_q` (submitted time is the earliest) that is marked as running is typically the JupyterHub session. Do not remove this job as it will disconnect your session.

If you want to access the working node at which the job is running, use the following command:

.. code-block:: bash

    condor_ssh_to_job 5888825.0

The expected output is

.. code-block:: bash

    (base) [cpardoar@ui04 dyn_database]$ condor_ssh_to_job 5888825.0
    Welcome to slot1_4@td820.pic.es!
    Your condor job is running with pid(s) 5888825.0

In this working node, we can directly access the :code:`_condor_stdout` file and check the current status by looking at the printed output of our job. To exit the working node enter:

.. code-block:: bash

    (base) [cpardoar@ui04 dyn_database]$ exit
    logout
    Connection to condor-job.td820.pic.es closed.

***********************************
Running different jobs in parallel
***********************************

To take advantage of HTCondor, we show here how to submit and process multiple jobs in parallel. There is a specific way to do this with HTCondor using the :code:`parameter` argument in the .submit file.
For example, if we want to run our script :code:`/simulate_population_dyn.py` with two different values of :code:`h_c` (in this example :code:`h_c = 1.7` and :code:`1.9`), we can use the following approaches:

* We create 2 different jsons (let us call them :code:`test1.json` and :code:`test2.json`) with the values of :code:`h_c`. Then :code:`test1.json` looks like this:

  .. code-block:: bash

    (base) [cpardoar@gpu05 ~]$  cat test1.json
    {"h_c":1.7}

  And :code:`test2.json` reads

  .. code-block:: bash

    (base) [cpardoar@gpu05 ~]$  cat test2.json
    {"h_c":1.9}

  We next have to specify in the submit file the jsons that will be taken as arguments for the :code:`parameter_override` in our .py script. We also have to specify 2 different directories to not mix the outputs from both simulation. Adjusting the submit file accordingly, we thus arrive at:

  .. code-block:: bash

    universe        = vanilla
    executable      = wrapper.sh
    output          = OUTPUT_2/hello.out.$(Cluster).$(Process).txt
    error           = OUTPUT_2/hello.error.$(Cluster).$(Process).txt
    log             = OUTPUT_2/hello.log.$(Cluster).$(Process).txt
    arguments =  /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test1 /nfs/pic.es/user/c/cpardoar/test1.json
    queue
    arguments =  /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test2 /nfs/pic.es/user/c/cpardoar/test2.json
    queue

  In the wrapper, we also have to specify that the :code:`parameter_override` and :code:`output` arguments will take the values passed via the submit file. Again, we just need to change the last line of our wrapper file:

  .. code-block:: bash

    #!/bin/bash

    export PATH=/data/magnesia/software/anaconda3/bin:$PATH
    conda init bash
    source /data/magnesia/software/anaconda3/etc/profile.d/conda.sh
    conda activate /data/magnesia/software/anaconda3/envs/pop_syn
    python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn.py --output $1 --parameter_override $2

* Although this above approach works, there are two more optimal ways to pass arguments to the wrapper using the submit file:

  1. Using a loop over the arguments:

  .. code-block:: bash

      (base) [cpardoar@gpu05 ~]$ test_argument.submit
      universe        = vanilla
      executable      = wrapper.sh
      output          = OUTPUT_args/hello.out.$(Cluster).$(Process).txt
      error           = OUTPUT_args/hello.error.$(Cluster).$(Process).txt
      log             = OUTPUT_args/hello.log.$(Cluster).$(Process).txt
      Queue arguments from (
      /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test1 /nfs/pic.es/user/c/cpardoar/test1.json
      /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test2 /nfs/pic.es/user/c/cpardoar/test2.json
      )

  .. note::
      It is important to open the bracket after :code:`from` and jump to the next line and start the list of arguments in a new row. The last line should just contain the closed bracket.

  2. Using a text file:

  .. code-block:: bash

      (base) [cpardoar@gpu05 ~]$ test_argument_txt.submit
      universe        = vanilla
      executable      = wrapper.sh
      output          = OUTPUT_2/hello.out.$(Cluster).$(Process).txt
      error           = OUTPUT_2/hello.error.$(Cluster).$(Process).txt
      log             = OUTPUT_2/hello.log.$(Cluster).$(Process).txt
      Queue arguments from arguments.txt


  where the :code:`arguments.txt` file looks like this:

  .. code-block:: bash

      (base) [cpardoar@ui03 test_htcondor]$ cat arguments.txt
      /data/magnesia/common/test_htcondor/OUTPUT_args_txt/output_repo_test1 /nfs/pic.es/user/c/cpardoar/test1.json
      /data/magnesia/common/test_htcondor/OUTPUT_args_txt/output_repo_test2 /nfs/pic.es/user/c/cpardoar/test2.json


*******************************
Additional examples of HTCondor
*******************************

  * How to submit a simple job with HTCondor: https://hcc.unl.edu/docs/osg/a_simple_example_of_submitting_an_htcondor_job/
  * How to take advantage of HTCondor with the argument parameter: https://swc-osg-workshop.github.io/2017-05-17-JLAB/novice/DHTC/04a-ScalingUp-python.html
