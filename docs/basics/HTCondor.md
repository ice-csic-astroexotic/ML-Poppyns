# HTCondor

This page is primarily for MAGNESIA developers. We focus on explaining the file structure and main commands needed to
submit simulations using the HTCondor infrastructure at PIC. The documentation provided by PIC about HTCondor can be 
found [here](https://pwiki.pic.es/index.php?title=HTCondor_User_Guide).

## File locations

The main software repo is located in the folder `/data/magnesia/software`. Note that the repo was cloned in this 
location following PIC's guidelines: We should use the `software` folder for shared repositories. 
The repo was cloned in such a way that every MAGNESIA user has writing, execution and reading access. A user would, in
principle, be able to clone the repo in their own home directories, but this should be avoided due to limited disk 
space. 

!!! warning
    
    Before launching simulations from any location in the server, the variable `server_run` in the
    `config_simulator.py` file should be set to `True` to ensure that the correct path to the software modules is 
    set.

The scripts to create the necessary files to submit a job and to manage the simulations with HTCondor are found in the
`utilities` folder in the subdirectory `PIC_scripts` in our repo. These are:

* `PIC_generate_htcondor_submit.py` generates the necessary HTCondor files to launch jobs.
* `PIC_check_simulations.py` checks which simulations have failed from a previous HTCondor run.
* `PIC_generate_htcondor_failed.py` checks for failed simulations and generates the HTCondor files to launch them again.
* After the failed simulations have been relaunched and finished successfully, `PIC_manage_failed_simulation.py` can be
used to transfer the new output back to the original folders.

We use the folder `/data/magnesia/common` to store the HTCondor files as it is PIC's recommended location for 
storing intermediate data. This is also where we store our simulations and ML experiments on intermediate timescales. 

There is also a scratch folder in `/data/magnesia/scratch` which contains our `conda` environment. This folder also 
offers extra disk space where output files of each run could be saved. These will however be deleted after each run 
and would thus need to be transferred elsewhere if required. Below we explain how to transfer files from this scratch
directory.


## Useful commands in HTCondor

The main commands when using HTCondor are the following:

* `condor_submit file.submit`  Submit the job(s) specified inside `file.submit`.
* `condor_q`  Query the status of the submitted jobs.
* `condor_rm <id_job>`   Remove a job.
* `condor_q -const 'JobStatus == 5' -af HoldReason`  Output the reason why jobs are being held.
* `condor_ssh_to_job <id_job>`  Enter the working node where the job is running.
* `condor_submit -i test.sub`  Submit job(s) in interactive mode.


## Steps to run the dynamical simulations


### SSH sessions

You can access PIC either via local terminal with SSH or via the JupyterHub interface at [https://jupyter.pic.es/](https://jupyter.pic.es/). 
To log in via SSH, type the following command in a terminal:
```commandline
ssh user@ui.pic.es
```
where `user` is your PIC username. You will be prompted to enter your password.

Currently, access via UAB's wireless and Ethernet network does not allow SSH connections. In particular, PIC uses port 
22 which is blocked. As a result, a standard SSH connection from ICE cannot be established at the moment. However, we 
can establish an ssh connection through a terminal session within [https://jupyter.pic.es/](https://jupyter.pic.es/). 
A standard SSH connection from outside the ICE can however be readily established.

The software repository located in `/data/magnesia/software` should be used to execute large experiments. 
Any changes, developments or updates of the code itself should be done on personal laptops whenever possible. 
To update the repository on the PIC servers, we use `git pull`. 
To establish GitHub access to the repository for the first time follow these steps:

  1. Paste the following command during an active SSH session, substituting your GitHub email address  
    `ssh-keygen -t ed25519 -C "your_email@example.com"`.
    This creates a new SSH key, using the provided email as a label.
  2. When you are prompted to "Enter a file in which to save the key," enter 
    `/data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  3. Start the SSH-agent in the background by entering `eval "$(ssh-agent -s)"`.
  4. Add your SSH private key to the SSH-agent by entering 
     `ssh-add /data/magnesia/software/ssh_keys/your_lastname` substituting in your last name.
  5. Then follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to add your new SSH-key to GitHub. 

!!! note

    To `git pull` and `git push` at a later stage, steps 3 and 4 have to be executed every time when using a SSH session.

### Submit files

In order to submit jobs with HTCondor, we need two different scripts: an HTCondor submit file and a wrapper (see below). 
The former is the file needed to submit a job with HTCondor, while the latter takes care of executing python relevant 
commands as well as running our python scripts. 

The HTCondor submit file looks as follows:

```commandline
(base) [cpardoar@ui02 test_htcondor]$ cat test.submit
# The UNIVERSE defines an execution environment. You will always use VANILLA.

universe        = vanilla

# Executable is the program your job will run.
executable      = wrapper.sh

# Location of standard output, standard error, and log files
# that HTCondor returns from the remote host.
output          = OUTPUT/hello.out.$(Cluster).$(Process).txt
error           = OUTPUT/hello.error.$(Cluster).$(Process).txt
log             = OUTPUT/hello.log.$(Cluster).$(Process).txt

queue
```

In our case, the executable is a wrapper (explained below) in which we call the .py file. When submitting an HTCondor 
job, simulations are run on a remote host. To see the terminal output (stdout) or errors (stderr) arising during the 
execution, we save the details in the path specified in the output, log and error variables. The path 
`OUTPUT/hello.out.$(Cluster).$(Process).txt` is an example. You can choose the path that is most convenient for your 
purpose. In our example, we want the output to be saved in a folder called `OUTPUT`, in the same location as the submit 
file and with the name `hello.out.$(Cluster).$(Process).txt`. 

For example, we could change the path of the output to
```commandline
output = test.txt
```
and the output (whatever is printed in the terminal during the execution of the wrapper) will be saved in the same 
folder as the submit file with the name `test.txt`.

!!! note

    If you want the path to be as in the previous example (`OUTPUT/hello.out.$(Cluster).$(Process).txt`), you have to 
    first manually create the `OUTPUT` directory in the location of the submit file. If the directory is not created 
    first an error will arise. 

In the example above, `$(Cluster)` represents the cluster identifier and `$(Process)` the process identifier. 
We use the cluster ID and process ID to differentiate between different runs and jobs. `Queue` is the start "button".

### Wrappers

As noted above, our wrappers contain the relevant Python scripts. An example wrapper looks like this:

```commandline
(base) [cpardoar@ui02 test_htcondor]$ cat wrapper.sh
#!/bin/bash

# Set where the anaconda installation is located in order to be able to use conda commands.
export PATH=/data/astro/software/centos7/conda/mambaforge_4.14.0/bin:$PATH

# Initialize anaconda in the bash shell.
conda init bash

# It is recommended by anaconda to close and restart the terminal after conda init.
source /data/astro/software/centos7/conda/mambaforge_4.14.0/etc/profile.d/conda.sh

# Activate conda environment.
conda activate /data/magnesia/scratch/conda/envs/pop_syn

# We copy the mlpoppyns module in the working node to avoid problems with the path while running the simulations in the server.
cp -R /data/magnesia/software/MAGNESIA_population_synthesis/mlpoppyns

# Run the simulation specifying the path for the output.
python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn_test.py --output /data/magnesia/common/test_HTCondor
```

!!! warning

    We cannot save the output from the simulations in the same folder as our `software` repository as the remote host 
    does not have writing access in `software`. Thus, we save the output in the `common` folder.

### Submitting and removing jobs

In order to submit a job (= running the simulation on the server), we use
```commandline
(base) [cpardoar@ui02 test_htcondor]$ condor_submit test.submit
```
If successful, this will return
```commandline
Submitting job(s).
1 job(s) submitted to cluster 5889056.
```
To look at the status of the jobs that we have submitted, run
```commandline
(base) [cpardoar@ui02 test_htcondor]$ condor_q
```
with the expected output
```commandline
-- Schedd: submit01.pic.es : <193.109.174.82:9618?... @ 02/18/22 18:39:10
OWNER    BATCH_NAME     SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
cpardoar ID: 5888746   2/18 10:32      _      1      _      1 5888746.0
cpardoar ID: 5888825   2/18 16:13      _      1      _      1 5888825.0

Total for query: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
Total for cpardoar: 2 jobs; 0 completed, 0 removed, 0 idle, 2 running, 0 held, 0 suspended
Total for all users: 1081 jobs; 0 completed, 0 removed, 256 idle, 818 running, 7 held, 0 suspended
```

In this example, we have 2 jobs running. Note that if you have initiated a session at [https://jupyter.pic.es/](https://jupyter.pic.es/), 
this session will appear as a running job (typically the first one submitted). 
If we see that some jobs are IDLE, these are currently in HTCondor's queue and are waiting to be launched. 
If you see a job that is on HOLD, this might indicate that something went wrong.

To see what caused the jobs to be held, run the following command:
```commandline
condor_q -const 'JobStatus == 5' -af HoldReason
```
If you want to remove a job, first execute `condor_q` to search for the `job_id` of the job you want to remove. 
For example, with the output from `condor_q` above, removing the job with the ID 5888825.0 can be achieved by running:
```commandline
condor_rm 5888825.0
```

!!! warning

    The first job shown by `condor_q` (submit time is the earliest) that is marked as running is typically the 
    JupyterHub session. Do not remove this job as this would disconnect your session.

If you want to access the working node in which the job is running, use the following command:
```commandline
condor_ssh_to_job 5888825.0
```

The expected output is
```commandline
(base) [cpardoar@ui04 dyn_database]$ condor_ssh_to_job 5888825.0
Welcome to slot1_4@td820.pic.es!
Your condor job is running with pid(s) 5888825.0
```

In this working node, we can directly access the `_condor_stdout` file and check the current status by looking at the 
printed output of our job. To exit the working node enter
```commandline
(base) [cpardoar@ui04 dyn_database]$ exit
logout
Connection to condor-job.td820.pic.es closed.
```

### Running different jobs in parallel

To take advantage of HTCondor, we show how to submit and process multiple jobs in parallel. HTCondor allows 
you to do this using the arguments `parameter` in the `.submit` file. For instance, if we want to run our script
`simulate_population_dyn.py` with two different values of `h_c` (e.g., `h_c = 1.7` and `1.9`), we can pass these values via
JSON files.

First, we create two different JSON files, test1.json and test2.json, containing the respective values of h_c:
```commandline
(base) [cpardoar@gpu05 ~]$  cat test1.json
{"h_c":1.7}
```
and `test2.json` reads
```commandline
(base) [cpardoar@gpu05 ~]$  cat test2.json
{"h_c":1.9}
```
Next, we have to specify in the submit file that the json files will be taken as two choices for the 
`parameter_override` argument in our .py script. We also have to specify two different directories for our 
`output_dir` argument to not mix up the outputs from both simulation. Adjusting the submit file accordingly, 
we thus arrive at:
```commandline
universe        = vanilla
executable      = wrapper.sh
output          = OUTPUT_2/hello.out.$(Cluster).$(Process).txt
error           = OUTPUT_2/hello.error.$(Cluster).$(Process).txt
log             = OUTPUT_2/hello.log.$(Cluster).$(Process).txt
arguments =  /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test1 /nfs/pic.es/user/c/cpardoar/test1.json
queue
arguments =  /data/magnesia/common/test_htcondor/OUTPUT_args/output_repo_test2 /nfs/pic.es/user/c/cpardoar/test2.json
queue
```
Note that the order of the arguments matters. In particular, in the wrapper, we have to specify that the
`parameter_override` and `output` arguments will take the values passed via the submit file. To do so, we again 
change the last line of our wrapper file:
```commandline
#!/bin/bash

export PATH=/data/magnesia/software/anaconda3/bin:$PATH
conda init bash
source /data/magnesia/software/anaconda3/etc/profile.d/conda.sh
conda activate /data/magnesia/software/anaconda3/envs/pop_syn
python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn.py --output $1 --parameter_override $
```

Although this above approach works, there are two more optimal ways to pass arguments to the wrapper using the
submit file:

* We can also use a loop over the arguments and modify the submit file as follows
   ```commandline
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
   ```

    !!! warning 

        It is important to open the bracket after `from` and jump to the next line and start the list of arguments in
        a new row. The last line should just contain the closed bracket.

  * We can also use a text file to pass the arguments. The submit file then reads
        ```commandline
        (base) [cpardoar@gpu05 ~]$ test_argument_txt.submit
        universe        = vanilla
        executable      = wrapper.sh
        output          = OUTPUT_2/hello.out.$(Cluster).$(Process).txt
        error           = OUTPUT_2/hello.error.$(Cluster).$(Process).txt
        log             = OUTPUT_2/hello.log.$(Cluster).$(Process).txt
        Queue arguments from arguments.txt
        ```
        where the `arguments.txt` file looks like this:
        ```commandline
        (base) [cpardoar@ui03 test_htcondor]$ cat arguments.txt
        /data/magnesia/common/test_htcondor/OUTPUT_args_txt/output_repo_test1 /nfs/pic.es/user/c/cpardoar/test1.json
        /data/magnesia/common/test_htcondor/OUTPUT_args_txt/output_repo_test2 /nfs/pic.es/user/c/cpardoar/test2.json
        ``` 
  
!!! warning

    If the output folders do not have write permissions for all users, the job will be held. To avoid this, 
    ensure that the output directories have the correct permissions by `chmod -R a+rwx <output_folder>`
  
### Additional HTCondor examples

  * [How to submit a simple job with HTCondor](https://hcc.unl.edu/docs/osg/a_simple_example_of_submitting_an_htcondor_job/)
  * [How to take advantage of HTCondor with the argument parameter](https://swc-osg-workshop.github.io/2017-05-17-JLAB/novice/DHTC/04a-ScalingUp-python.html)
