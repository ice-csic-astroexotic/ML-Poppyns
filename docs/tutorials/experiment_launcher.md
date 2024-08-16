# Experiment launcher

The `utilities/experiment_helper/experiment_launcher.py` script allows us to execute a list of experiment commands 
specified in a text file. For example, to launch the generator script with different arguments, this text file could 
contain the following content:
```commandline
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_dataset/array_64 --type array --resolution_dyn 64 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_dataset/array_128 --type array --resolution_dyn 128 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_dataset/array_256 --type array --resolution_dyn 256 --resolution_ppdot 32
python  pypopsyn/generator/generate_dataset_full.py --data simulated_data --save_dir generated_dataset/array_512 --type array --resolution_dyn 512 --resolution_ppdot 32
```
  
We can also use it to launch different training experiments. In this case, the text file could look like this:
```commandline
python pypopsyn/learning/train.py --dataset_training generated_dataset/array_64/train_dataset.csv --dataset_validation generated_dataset/array_64/dataset_valid.csv --dataset_statistics generated_dataset/array_64/statistics_train.json --input_shape 4 64 64 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r64_gc_position_velocity
python pypopsyn/learning/train.py --dataset_training generated_dataset/array_128/train_dataset.csv --dataset_validation generated_dataset/array_128/dataset_valid.csv --dataset_statistics generated_dataset/array_128/statistics_train.json --input_shape 4 128 128 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r128_gc_position_velocity
python pypopsyn/learning/train.py --dataset_training generated_dataset/array_256/train_dataset.csv --dataset_validation generated_dataset/array_256/dataset_valid.csv --dataset_statistics generated_dataset/array_256/statistics_train.json --input_shape 4 256 256 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r256_gc_position_velocity
python pypopsyn/learning/train.py --dataset_training generated_dataset/array_512/train_dataset.csv --dataset_validation generated_dataset/array_512/dataset_valid.csv --dataset_statistics generated_dataset/array_512/statistics_train.json --input_shape 4 512 512 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r512_gc_position_velocity
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
