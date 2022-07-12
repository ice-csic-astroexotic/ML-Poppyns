################################################
HTCondor
################################################

In this document we will focus on explaining the structure of the files and the main commands needed to submit the simulations in the HTCondor infrastructure at PIC. The documentation provided by PIC about HTCondor is here: https://pwiki.pic.es/index.php?title=HTCondor_User_Guide

**********************
Location of the files
**********************

#. The repo is in the folder :code:`/data/magnesia/software`. Note that the repo was cloned here following the PIC guidelines: we should use the software folder for shared repositories. The repo was cloned in such a way that every user of magnesia has writing, execution and reading access. A user would be able to clone the repo in their own home directories, but this should be avoided due to limited disk space.
#. The scripts to submit a job in HTCondor are in the common folder :code:`/data/magnesia/common`. We use this folder to store the HTCondor files since it is the recommended folder by PIC to store intermediate data.
#. There is a scratch folder in the magnesia disk space where the output files of each run could be saved. These will however be deleted after each run and thus need to be transferred elsewhere if required. Below we explain how to transfer files from this scratch directory.

***************************
Useful commands in HTCondor
***************************

Here I list the commands explained below:

#. :code:`condor_submit file.submit`  Submit the job specified inside file.submit.
#. :code:`condor_q`  Status of the submitted jobs.
#. :code:`condor_rm <id_job>`   Remove a job.
#. :code:`condor_q -const 'JobStatus == 5' -af HoldReason`  Output the reason why some of the jobs are being held.
#. :code:`condor_ssh_to_job <id_job>`  Enter the working node where the job is running.
#. :code:`condor_submit -i test.sub`  Submit a job in interactive mode.

Steps to run the dynamical simulations
**************************************

* Via the JupyterHub online interface at https://jupyter.pic.es/ we are able to run and read files in the software folder. However, to have writing access to this folder we need to log in via ssh. Then type the following command in the terminal:

  .. code-block:: bash

    ssh user@ui.pic.es

  where :code:`user` is your PIC user name. You will be prompted to enter your password.
  Currently access via UAB's wireless and ethernet network does not allow ssh connections; in particular PIC uses port 22 which is blocked. As a result a standard ssh connection from the ICE cannot be established at the moment.  However, we can establish a connection through the terminal of https://jupyter.pic.es/ and from here we should be able to ssh in. A standard ssh connection from outside the ICE can however be readily established.

* This repository should be used to execute large experiments. Any change or update of the code should be done on our personal laptops. To update the repository on the server we use git pull. To give GitHub access to the repository follow these steps:

  #. Paste the text below, substituting in your GitHub email address  :code:`ssh-keygen -t ed25519 -C "your_email@example.com"` This creates a new SSH key, using the provided email as a label.
  #. When you're prompted to "Enter a file in which to save the key," enter :code:`/data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  #. Start the ssh-agent in the background by entering :code:`eval "$(ssh-agent -s)"`.
  #. Add your SSH private key to the ssh-agent by entering :code:`ssh-add /data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  #. Then follow these instructions to add your new ssh-key to GitHub: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

  .. note::
    To :code:`git pull` and :code:`git push` steps 3 and 4 should be done every time that we log in to the server.

* In  order to submit jobs with HTCondor we need two different scripts: an HTCondor submit file and a wrapper (see below). The former is the file needed to submit a job in HTCondor and the latter takes care of executing python relevant commands as well as running our python scripts. These two scripts are saved in :code:`/data/magnesia/common/test_htcondor`.

  * The HTCondor submit file looks like the following:

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


  In our case the executable is a wrapper (explained below) where we call the .py file. Using  HTCondor the simulations are run in a remote host. To see what is printed in the terminal (stdout) or errors arising during the execution (stderr) we save the details in the path specified in the output, log and error variables. The path :code:`OUTPUT/hello.out.$(Cluster).$(Process).txt` is an example. You can choose the path that is more convenient for you. In this case we want the output to be saved in a folder called :code:`OUTPUT` in the same folder where the submit file is located and with the name :code:`hello.out.$(Cluster).$(Process).txt`. For example, we could change the path of the output to

  .. code-block:: bash

    output = test.txt

  and the output (what is printed in the terminal during the execution of the wrapper) will be saved in the same folder as the submit file with the name :code:`test.txt`. Note that if we want the path to be as in the previous example (:code:`OUTPUT/hello.out.$(Cluster).$(Process).txt`) we have to first create the :code:`OUTPUT` directory where the submit file is; if it is not created an error will arise. In the script above :code:`$(Cluster)` means the cluster identifier and :code:`$(Process)` the process identifier. We use the cluster id and process id to differentiate between different runs and jobs. :code:`Queue` is the start "button".

  .. note::
    Files saved in the scratch directory are removed after the run. To transfer these files to a directory we need to specify it in the submit file.

  For example, if the output file is called :code:`output1.txt` and we want to keep that file we need to add to the submit file the following line:

  .. code-block:: bash

    transfer_output_files= output1.txt

  Then :code:`output1.txt` will be saved in the same directory as the submit file.

  * The wrapper looks like this:

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
    We can't save the output from the simulations in the same folder as the repo since the remote host doesn't have writing access. Thus, we save the output in the common folder.

* In order to submit a job (= running the simulation on the server) we use:

  .. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ condor_submit test.submit

  and it should return:

  .. code-block:: bash

    Submitting job(s).
    1 job(s) submitted to cluster 5889056.

* To look at the status of the jobs that we have submitted run:

  .. code-block:: bash

    (base) [cpardoar@ui02 test_htcondor]$ condor_q

  with the expected output:

  .. code-block:: bash

    -- Schedd: submit01.pic.es : <193.109.174.82:9618?... @ 02/18/22 18:39:10
    OWNER    BATCH_NAME     SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
    cpardoar ID: 5888746   2/18 10:32      _      1      _      1 5888746.0
    cpardoar ID: 5888825   2/18 16:13      _      1      _      1 5888825.0

    Total for query: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
    Total for cpardoar: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
    Total for all users: 1081 jobs; 0 completed, 0 removed, 256 idle, 818 running, 7 held, 0 suspended

  Here it says that we have 2 jobs running. If we are using the terminal of https://jupyter.pic.es/ you will have one job running which is the jupyter notebook. If we see that the jobs are  IDLE it means that due to how HTCondor manages the queue, we are waiting for our job to be run.
  If we see a job that is on hold this might indicate that something went wrong. To see what happened to held jobs run the following command:

  .. code-block:: bash

    condor_q -const 'JobStatus == 5' -af HoldReason

  If we want to remove any job, then we execute :code:`condor_q` to search for the :code:`job_id` that we want to remove. For example with the output from :code:`condor_q` above if we want to remove the job with id 5888825.0 we would do the following:

  .. code-block:: bash

    condor_rm 5888825.0

  .. note::
    The first job in the  :code:`condor_q` output (the submitted time is always the earliest) that appears running is the jupyter notebook. Do not remove this one because you will close the jupyter session.

  If we want to enter the working node where the job is running we can use the following command:

  .. code-block:: bash

    condor_ssh_to_job 5888825.0

  the expected output is:

  .. code-block:: bash

    (base) [cpardoar@ui04 dyn_database]$ condor_ssh_to_job 5888825.0
    Welcome to slot1_4@td820.pic.es!
    Your condor job is running with pid(s) 5888825.0

  In this working node we can see the :code:`_condor_stdout` file where we can check the current status by looking at the printed output of our job. To exit from the working node enter:

  .. code-block:: bash

    (base) [cpardoar@ui04 dyn_database]$ exit
    logout
    Connection to condor-job.td820.pic.es closed.

***********************************
Running different jobs in parallel
***********************************

To take advantage of HTCondor we show here how to submit and process multiple jobs in parallel. There is a specific way to do this in HTCondor using the parameter "argument" in the .submit file.
For example if we want to run our :code:`/simulate_population_dyn.py` with two different values of :code:`h_c` (in this example :code:`h_c` = 1.7 and 1.9). We should do the following steps:

* We create 2 different jsons (let us call them :code:`test1.json` and :code:`test2.json`) with the values of :code:`h_c`. Then :code:`test1.json` looks like:

  .. code-block:: bash

    (base) [cpardoar@gpu05 ~]$  cat test1.json
    {"h_c":1.7}

  and

  .. code-block:: bash

    (base) [cpardoar@gpu05 ~]$  cat test2.json
    {"h_c":1.9}

* Then we have to specify in the submit file the jsons that will be taken as an argument for the :code:`parameter_override`. Also we have to specify 2 different directories to not mix the output from the simulation. Thus, we have that our submit file is the same as before apart from the last lines:

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

* In the wrapper we also have to specify that the :code:`parameter_override` and output will take the values passed via the submit file. Again we just need to change the last line of our wrapper file:

  .. code-block:: bash

    #!/bin/bash

    export PATH=/data/magnesia/software/anaconda3/bin:$PATH
    conda init bash
    source /data/magnesia/software/anaconda3/etc/profile.d/conda.sh
    conda activate /data/magnesia/software/anaconda3/envs/pop_syn
    python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn.py --output $1 --parameter_override $2


  You can pass these arguments to the wrapper in the submit file in 2 more optimal ways:

  * Using a loop over the arguments:

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
      It is important to open the bracket after the from and jump to the next line and start the list of arguments in a new row. The last line should just contain the closed bracket.

  * Using a txt file:

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


Examples of HTCondor
********************
  * How to submit a simple job in HTCondor: https://hcc.unl.edu/docs/osg/a_simple_example_of_submitting_an_htcondor_job/
  * How to take advantage of HTCondor with the argument parameter: https://swc-osg-workshop.github.io/2017-05-17-JLAB/novice/DHTC/04a-ScalingUp-python.html
