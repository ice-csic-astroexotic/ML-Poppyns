"""
    Generation of the HTCondor scripts.

    We generate all the scripts needed to run the whole set of simulations in chunks using HTCondor.
    Each job will run each chunk of simulations for a different set of initial parameters.

    Note that this script needs the output from the `parameter_sweeper.py` script.
    The submit files for each job, the arguments.txt and a wrapper are saved in the path specified with the command
    line argument --output_dir_htcondor.


    Authors:
        Celsa Pardo (pardo @ csic.es)


Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import pathlib

import numpy as np

from pypopsyn.simulator.configuration import cfg


def generate_job_submit(
    path_output: pathlib.Path, path_arguments: pathlib.Path
):
    """
    Create all the submit files.

    Args:
        path_output (pathlib.Path): Output directory for the submit file.
        path_arguments (pathlib.Path): Path for the simulation arguments.

    Returns:
        Nothing.
    """
    # Path where each submit file will be saved.
    path_submit = pathlib.Path().joinpath(path_output, "job.submit")

    # Writing the HTCondor submit file, specifying the relevant arguments, output/error paths and queue structure.

    with open(path_submit, "w") as f:
        f.write("universe        = vanilla \n")
        f.write("executable      = " + str(path_output) + "/wrapper.sh \n")
        f.write(
            "output          = " + str(path_output) + "/$(ProcId)-out.txt \n"
        )
        f.write(
            "error           = " + str(path_output) + "/$(ProcId)-error.txt \n"
        )
        f.write("Queue arguments from " + str(path_arguments) + "\n")
        f.close()


def generate_wrapper(
    type_simulation: pathlib.Path,
    dyn_path: pathlib.Path,
    path_wrapper: pathlib.Path,
):

    """
    Create all the wrapper files.

    Args:

        type_simulation (str): String with the type of simulation we want to run.
        path_wrapper (pathlib.Path): Output directory for the wrapper file.
        dyn_path (pathlib.Path): Path to where the dynamically evolved population database is stored.


    Returns:

        Nothing.
    """

    # Writing the `wrapper.sh` file where we loop over the lines of the `"/arguments_job" + str(j + 1) + ".txt"` file.

    if type_simulation == "dyn":

        exec_command = "python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn.py --output_dir ${a[0]} --parameter_override ${a[1]}  \n"

    elif type_simulation == "magrot":

        exec_command = (
            "python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_magrot_det.py --dyn_data "
            + str(dyn_path)
            + " --output_dir ${a[0]} --parameter_override ${a[1]} \n"
        )
    else:
        raise ValueError(
            "The specified simulation type is not feasible, choose between dyn or magrot."
        )

    with open(path_wrapper, "w") as f:
        f.write("#!/bin/bash \n")
        f.write("\n")
        f.write(
            "export PATH=/data/astro/software/centos7/conda/mambaforge_4.14.0/bin:$PATH\n"
        )
        f.write("conda init bash\n")

        f.write(
            "source /data/astro/software/centos7/conda/mambaforge_4.14.0/etc/profile.d/conda.sh\n"
        )
        f.write(
            "conda activate /data/magnesia/scratch/conda/env/pop_syn_test\n"
        )
        f.write(
            "cp -R /data/magnesia/software/MAGNESIA_population_synthesis/pypopsyn ."
        )
        f.write("filename=$1 \n")
        f.write(" while read line; do \n")
        f.write("# Reading each line. \n")
        f.write("myarr[$index]=$line \n")
        f.write("#Extract each element from the lines. \n")
        f.write("a=(${myarr[$index]})\n")
        f.write(exec_command)
        f.write("done < $filename\n")
        f.write("conda deactivate")
        f.close()


def submit_generator(args):

    common_path = cfg["path_server_output"]

    simulation_output_path = pathlib.Path(args.output_dir_simulation)

    output_htcondor_path = pathlib.Path(args.output_dir_htcondor)

    output_htcondor_path.mkdir(parents=True, exist_ok=True)

    # Reading the txt generated by the ´parameter_sweeper.py´ that contains the simulation arguments.
    simulation_arguments = np.genfromtxt(
        pathlib.Path().joinpath(
            simulation_output_path, "simulation_arguments.txt"
        ),
        dtype=str,
    )

    n_sim_total = len(simulation_arguments)

    # Determine the number of weeks required to run all the simulations, where we assume args.n_sim_week simulations
    # can be run each week.
    if n_sim_total % args.n_sim_week == 0:
        n_week = int(n_sim_total / args.n_sim_week)
    else:
        n_week = int(n_sim_total / args.n_sim_week) + 1

    for i in range(n_week):

        # We create one folder per week.
        week_folder_path = pathlib.Path().joinpath(
            output_htcondor_path, "week-" + str(i)
        )
        week_folder_path.mkdir(parents=True, exist_ok=True)

        # Taking the chunk of simulations that we want to run this week.
        sim_week = simulation_arguments[
            i * args.n_sim_week : (i + 1) * args.n_sim_week
        ]

        # We need to count how many simulations we have to run in each week, since if we have that
        # ´n_sim_total % args.n_sim_week != 0´ then we will end up with fewer simulations than args.n_sim_week
        # in the last week.

        n_sim_folder = len(sim_week)

        # Calculate the number of ´argument_job<j>.txt´ files we need in order to have ´args.n_sim_job´ simulations
        # per job, i.e., ´args.n_sim_job´ lines in each ´argument_job<j>.txt´ .

        if n_sim_folder % args.n_sim_job == 0:
            n_args = int(n_sim_folder / args.n_sim_job)
        else:
            n_args = int(n_sim_folder / args.n_sim_job) + 1

        # Create the ´wrapper.sh´.
        path_wrapper = pathlib.Path().joinpath(week_folder_path, "wrapper.sh")

        generate_wrapper(args.type_simulation, args.dyn_data, path_wrapper)

        # Create the ´.submit ´ file.
        path_arguments = pathlib.Path().joinpath(
            week_folder_path, "arguments_week.txt"
        )
        generate_job_submit(week_folder_path, path_arguments)

        list_arguments_week = []

        # Looping through the `n_args` to create each `"/arguments_job" + str(j + 1) + ".txt"` file.
        for j in range(n_args):

            chunk_sim_job = sim_week[
                j * args.n_sim_job : (j + 1) * args.n_sim_job
            ]

            path_arguments_job = pathlib.Path().joinpath(
                common_path,
                str(week_folder_path) + "/arguments_job" + str(j + 1) + ".txt",
            )

            np.savetxt(path_arguments_job, chunk_sim_job, fmt="%s")

            # List of all the paths of each `"/arguments_job" + str(j + 1) + ".txt"` file
            list_arguments_week.append(path_arguments_job)

        np.savetxt(path_arguments, list_arguments_week, fmt="%s")


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="HTCondor parameter")

    args.add_argument(
        "--output_dir_htcondor",
        nargs="?",
        type=str,
        default="output_htcondor",
        help="Path to the directory where the files generated in this script will be saved.",
    )

    args.add_argument(
        "--output_dir_simulation",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory where the folders created with the parameter_sweeper.py are.",
    )

    args.add_argument(
        "--n_sim_job",
        nargs="?",
        type=int,
        default=150,
        help="Number of simulations per job.",
    )

    args.add_argument(
        "--n_sim_week",
        nargs="?",
        type=int,
        default=False,
        help="Number of simulations per week.",
    )

    args.add_argument(
        "--dyn_data",
        nargs="?",
        type=str,
        default=None,
        help="If using the simulator simulate_population_magrot_det, path to the file where "
        "the dynamically evolved population database is stored.",
    )

    args.add_argument(
        "--type_simulation",
        nargs="?",
        type=str,
        default=None,
        help="Type of simulation that we want to run in the PIC with HTCondor. Choose between dyn or magrot.",
    )

    args = args.parse_args()

    submit_generator(args)
