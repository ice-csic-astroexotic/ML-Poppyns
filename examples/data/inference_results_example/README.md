Inference results 

## Description
Inference results of the Convolutional neural network trained respectively to predict either 1) the `sigma_k` parameter of the maxwell kick velocity or 2) the scale height `h_c` of the birth Galactic height distribution or 3) both parameters simultaneusly.
The network have been tested on the validation datasets simulated with 1) a varying `sigma_k` parameter, 2) a varying `h_c` parameter and 3) both varying `sigma_k` and `h_c` respectively.
The total datasets for experiment 1) and 2) contains 20000 simulated samples each, while for experiment 3) a dataset of 16384 simulated samples has been used. 
A 80-20 % training-validation split has been adopted for all the experiments.
We use a 3-channel input with 1 density map and 2 proper motion maps in ICRS frame with resolution 128 x 64.
The trained networks and the datasets used are saved on the PIC server.
The inference results are reported here to show the usage of the inference script.

The inference results for the three experiment 1), 2) and 3) are saved into .csv files.

## Instructions to Run the inference script

commit hash: 93276ffb3a4bd6a479feeceb13ba2ed5f63c8c72

1) Infer only `sigma_k`:
```
python examples/learning/infer.py --configuration examples/learning/config_multiparameter_CNN.json --weights /data/magnesia/common/learning/maxwell_kick/icrs_position_velocity/cnn/128_resolution/s20000_r128_icrs_position_velocity/models/Convolution/1127_175921/best_model_trial1.pth --dataset /data/magnesia/common/generated_datasets/maxwell_kick/128_resolution/array_20000_samples_128_res/valid_dataset.csv --ignored_inputs 0 1 3 4 5 --ignored_labels 8 --input_shape 3 128 64 --num_parameters 1 --normalize 1 --save_dir inference_results
```

2) Infer only `h_c`:
```
python examples/learning/infer.py --configuration examples/learning/config_multiparameter_CNN.json --weights /data/magnesia/common/learning/z_position/icrs_position_velocity/cnn/128_resolution/s20000_r128_icrs_position_velocity/models/Convolution/1129_100952/best_model_trial1.pth --dataset /data/magnesia/common/generated_datasets/z_position/128_resolution/array_20000_samples_128_res/valid_dataset.csv --ignored_inputs 0 1 3 4 5 --ignored_labels 9 --input_shape 3 128 64 --num_parameters 1 --normalize 1 --save_dir inference_results
```

3) Infer both `sigma_k` and `h_c`:
```
python examples/learning/infer.py --configuration examples/learning/config_multiparameter_CNN.json --weights /data/magnesia/common/learning/maxwell_kick_plus_z_position/icrs_position_velocity/cnn/128_resolution/s128x128_r128_icrs_position_velocity/models/Convolution/1127_100653/best_model_trial1.pth --dataset /data/magnesia/common/generated_datasets/maxwell_kick_plus_z_position/128_resolution/array_128times128_samples_128_res/valid_dataset.csv --ignored_inputs 0 1 3 4 5 --input_shape 3 128 64 --num_parameters 2 --normalize 1 --save_dir inference_results
```
