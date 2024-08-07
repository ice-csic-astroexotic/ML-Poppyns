# Experiment launcher

The `utilities/experiment_helper/experiment_launcher.py` script allows us to execute a list of experiment commands 
specified in a text file. For example, this text file could contain the following content:
```commandline
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_64 --type array --resolution_dyn 64 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_128 --type array --resolution_dyn 128 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_256 --type array --resolution_dyn 256 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_512 --type array --resolution_dyn 512 --resolution_ppdot 32
```
  
Each line should contain one full command (including the `python` program call) to execute a given experiment. By 
default, the command list will be stored in a file called `command_list.txt`. However, a custom experiments file can 
also be specified with the `--command_list` parameter as follows:
```commandline
python utilities/experiment_helper/experiment_launcher.py --command_list pypopsyn/generator/experiment_list.txt --processes 2
```
The script will then execute all experiments outlined in the text file automatically. In the above example, we have 
set the `--processes` argument to two, which allows us to have a maximum of two processes running in parallel. 
Note that this number should reflect the number of available threads/cores.

This module combined with the intelligent use of the `--save_dir` `CLI` argument as shown in the example above allows 
us to run many experiments unattended and check them asynchronously.
