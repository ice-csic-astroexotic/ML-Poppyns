"""
    Generating HTCondor files for failed simulations at the PIC.

    With this script, we create the structure needed to rerun the simulations
    that have failed when running the whole set of simulations using HTCondor.
    Note that in this case, we run each simulation one by one,
    i.e., one simulation per job to prevent MaxwallTime problems.

    Note that to run this script, we first need to run the check_simulation.py script
    to generate the failed_folders.csv file.

    Running the code:

        python PIC_generate_htcondor_failed.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo @ csic.es)
"""

import argparse
import pathlib
import shutil

import numpy as np
import pandas as pd


def generate_htcondor_failed(args):
    """
    Create the submit file and wrapper file necessary to relaunch the failed simulations.

    Args:
        args:
            output_dir_simulation (pathlib.Path): Output directory where the simulation outputs are.
            number_sim_job (int): Number of simulations per job were chosen when running the original set of
                simulations with HTCondor.
            dyn_data (pathlib.Path): Path to the file where the dynamically evolved population database is stored.
            type_simulation (str): Type of simulation that we want to run in the PIC with HTCondor.
                Choose between dyn or magrot.

    Returns:
        Nothing.
    """
    output_simulations_path = pathlib.Path(args.output_dir_simulation)

    # If we want to copy and check the error.txt and out.txt of the failed simulations from HTCondor
    # in order to assess the reason for failure, the following variable should be set to True.
    check_simulations_failed = False

    # Reading the `failed_folders.csv` file, where the simulations that have failed are saved using the
    # `PIC_check_simulations.py` script.

    data_failed_folder = pd.read_csv(
        pathlib.Path().joinpath(
            args.dir_failed_folder_csv, "failed_folders.csv"
        )
    )
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

        # Copying the err.txt and out.txt files of the failed simulations from the htcondor_output folder
        # if this is needed to check why the simulations have failed.
        # Note that each of the out.txt and err.txt files contain _stdout and _sterr from more than one simulation.
        # Specifically, there will be as many as the args.number_sim_job.

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

    # Creating the new submit and argument.txt files for relaunching the failed simulations.
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

    # Writing the HTCondor submit files, which specify the relevant arguments, output/error paths and queue structure.
    with open(failed_submit_path, "w") as f:
        f.write("universe        = vanilla \n")
        f.write("executable      = " + str(failed_wrapper_path) + "\n")
        f.write("arguments       = $(arg1) $(arg2) \n")
        f.write("output          = $(arg1)/out.txt \n")
        f.write("error           = $(arg1)/error.txt \n")
        f.write("log             = $(arg1)/log.txt \n")
        f.write("Queue arg1 arg2 from " + str(failed_arguments_path) + "\n")
        f.close()

    # We will create the wrapper file according to the args.type_simulation argument. For example,
    # if it is equal to 'dyn', we generate a wrapper file that will execute the dynamical simulations.
    if args.type_simulation == "dyn":

        exec_command = "python /data/magnesia/software/MAGNESIA_population_synthesis/pypopsyn/simulator/simulate_population_dyn.py --output_dir $1 --parameter_override $2  \n"

    elif args.type_simulation == "magrot":

        exec_command = (
            "python /data/magnesia/software/MAGNESIA_population_synthesis/pypopsyn/simulator/simulate_population_magrot_det.py --dyn_data "
            + str(args.dyn_data)
            + " --output_dir $1 --parameter_override $2 \n"
        )
    else:
        raise ValueError(
            "The specified simulation type is not feasible. Choose between dyn or magrot."
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
        f.write("conda activate /data/magnesia/scratch2/conda/envs/pop_syn\n")
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
        default=None,
        required=True,
        help="Number of simulations per job were chosen when running the original set of simulations with HTCondor.",
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

    args.add_argument(
        "--dir_failed_folder_csv",
        nargs="?",
        type=str,
        default="output/test",
        help="Path to the directory where the csv file containing the list of failed folders is saved.",
    )

    args = args.parse_args()
    generate_htcondor_failed(args)
