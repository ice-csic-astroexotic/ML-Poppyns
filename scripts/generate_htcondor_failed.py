"""

With this script we create the structure needed to rerun in HTCondor
the simulations that have failed when running the whole set of simulations.
Note that in this case we run each simulation one by one, i.e. one simulation per job to prevent MaxwallTime problems.

Note that to run this scripts first we need to run the check_simulation.py script to generate failed_folders.csv file.

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
import shutil

import numpy as np
import pandas as pd


def generate_htcondor_failed(args):

    output_simulations_path = pathlib.Path(args.output_dir_simulation)

    # If we want to check the error.txt and out.txt of the failed simulations from HTCondor we need to set up this variable to True.
    check_simulations_failed = False

    data_failed_folder = pd.read_csv("scripts/failed_folders.csv")
    list_failed_folder = data_failed_folder["folder"]

    directory_path = output_simulations_path.parents[0]
    failed_simulation_path = pathlib.Path().joinpath(
        directory_path, "failed_simulations"
    )
    output_failed_simulations = pathlib.Path().joinpath(
        failed_simulation_path, "output_simulations"
    )
    htcondor_failed_submit_path = pathlib.Path().joinpath(
        failed_simulation_path, "htcondor_submit"
    )
    htcondor_failed_output_path = pathlib.Path().joinpath(
        failed_simulation_path, "htcondor_output"
    )

    failed_simulation_path.mkdir(exist_ok=True)
    output_failed_simulations.mkdir(exist_ok=True)
    htcondor_failed_submit_path.mkdir(exist_ok=True)
    htcondor_failed_output_path.mkdir(exist_ok=True)

    for i, simulation in enumerate(list_failed_folder):

        simulation = f"{simulation:06}"
        # Copy the folder with the failed simulations output to the failed_simulation folder.
        simulation_folder = pathlib.Path().joinpath(
            output_simulations_path, simulation
        )
        shutil.copytree(
            simulation_folder,
            pathlib.Path().joinpath(output_failed_simulations, simulation),
            dirs_exist_ok=True,
        )

        # Copying the err.txt and out.txt files of the failed simulations from the htcondor_output folder if it is needed to check why the simulations have failed.
        # Note that we have that in each of the out.txt and err.txt files there are the _stdout and _sterr from more than one simulation, it will be as many as the args.number_sim_job.

        if check_simulations_failed:
            out_txt_number = int(int(simulation) / args.number_sim_job) + 1

            htcondor_output_path = pathlib.Path().joinpath(
                directory_path, "HTCondor_output"
            )

            out_txt_path = pathlib.Path().joinpath(
                htcondor_output_path, str(out_txt_number) + "-out.txt"
            )
            err_txt_path = pathlib.Path().joinpath(
                htcondor_output_path, str(out_txt_number) + "-error.txt"
            )
            out_failed_txt_path = pathlib.Path().joinpath(
                htcondor_failed_output_path, str(out_txt_number) + "-out.txt"
            )
            err_failed_txt_path = pathlib.Path().joinpath(
                htcondor_failed_output_path, str(out_txt_number) + "-error.txt"
            )

            shutil.copy(out_txt_path, out_failed_txt_path)
            shutil.copy(err_txt_path, err_failed_txt_path)

    # Creating the new submit, argument.txt files for relauching the failed simulations
    failed_arguments_path = pathlib.Path().joinpath(
        htcondor_failed_submit_path, "arguments_.txt"
    )
    failed_submit_path = pathlib.Path().joinpath(
        htcondor_failed_submit_path, "job.submit"
    )
    failed_wrapper_path = pathlib.Path().joinpath(
        htcondor_failed_submit_path, "wrapper.sh"
    )

    list_arguments = [
        str(output_failed_simulations)
        + "/"
        + f"{path:06}"
        + " "
        + str(output_failed_simulations)
        + "/"
        + f"{path:06}"
        + "/override.json"
        for path in list_failed_folder
    ]
    np.savetxt(failed_arguments_path, list_arguments, fmt="%s")

    # Writing the HTCondor submit file, specifying the relevant arguments, output/error paths and queue structure.
    with open(failed_submit_path, "w") as f:
        f.write("universe        = vanilla \n")
        f.write("executable      = " + str(failed_wrapper_path) + "\n")
        f.write("arguments       = $(arg1) $(arg2) \n")
        f.write("output          = $(arg1)/out.txt \n")
        f.write("error           = $(arg1)/error.txt \n")
        f.write("log             = $(arg1)/log.txt \n")
        f.write("Queue arg1 arg2 from " + str(failed_arguments_path) + "\n")
        f.close()

    if args.type_simulation == "dyn":

        exec_command = "python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_dyn.py --output_dir ${a[0]} --parameter_override ${a[1]}  \n"

    elif args.type_simulation == "magrot":

        exec_command = (
            "python /data/magnesia/software/MAGNESIA_population_synthesis/examples/simulator/simulate_population_magrot_det.py --dyn_data "
            + str(args.dyn_path)
            + " --output_dir ${a[0]} --parameter_override ${a[1]} \n"
        )
    else:
        raise ValueError(
            "The specified simulation type is not feasible, choose between dyn or magrot."
        )

    with open(failed_wrapper_path, "w") as f:
        f.write("#!/bin/bash \n")
        f.write("\n")
        f.write(
            "export PATH=/data/astro/software/centos7/conda/mambaforge_4.14.0/bin:$PATH\n"
        )
        f.write("conda init bash\n")

        f.write(
            "source /data/astro/software/centos7/conda/mambaforge_4.14.0/etc/profile.d/conda.sh\n"
        )
        f.write("conda activate /data/magnesia/scratch/conda/env/pop_syn\n")
        f.write(
            "# We copy the pypopsyn module in the working node to avoid problems with the path while running the simulations in the server.\n"
        )
        f.write(
            "cp -R /data/magnesia/software/MAGNESIA_population_synthesis/pypopsyn .\n"
        )
        f.write(exec_command)
        f.write("conda deactivate")
        f.close()


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="HTCondor parameter")

    args.add_argument(
        "--output_dir_simulation",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory with the output from the simulations.",
    )

    args.add_argument(
        "--number_sim_job",
        nargs="?",
        type=int,
        default=30,
        help="Number of simulations per job that we chose when running the original set of simulations in HTCondor.",
    )

    args.add_argument(
        "--dyn_data",
        nargs="?",
        type=str,
        default="simulator/output",
        help="Path to the file where the dynamically evolved population database is stored.",
    )

    args.add_argument(
        "--type_simulation",
        nargs="?",
        type=str,
        default=None,
        help="Type of simulation that we want to run in the PIC with HTCondor. Choose between dyn or magrot.",
    )

    args = args.parse_args()
    generate_htcondor_failed(args)
