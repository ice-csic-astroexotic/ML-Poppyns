# Experiment launcher

The `utilities/experiment_helper/experiment_launcher.py` script allows you to specify a list of experiment commands in a text file like:
```commandline
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_64 --type array --resolution_dyn 64 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_128 --type array --resolution_dyn 128 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_256 --type array --resolution_dyn 256 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_data/array_512 --type array --resolution_dyn 512 --resolution_ppdot 32
```
  
By default, the command list will be held in `command_list.txt`. Each line should contain one full command (including the `python` program call) to execute an experiment. 
The script will execute those experiments automatically and in parallel providing a number of maximum simultaneous `--processes`.
Obviously, this number of processes should be set as a function of the number of available threads/cores.

A custom experiments file can be specified with the `--command_list` parameter:
```commandline
python utilities/experiment_helper/experiment_launcher.py --command_list pypopsyn/generator/experiment_list.txt --processes 2
```

This combined with an intelligent use of the `--save_dir` `CLI` argument will let you run many experiments unattended and check them asynchronously.
