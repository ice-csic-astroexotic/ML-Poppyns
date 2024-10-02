# Learning pulsar parameters with NNs

In the following, we describe how neural networks can be used to estimate the characteristic parameters that describe
the birth properties of a population of neutron stars. Here, we focus on obtaining point estimates of our parameters.
I.e., we do not derive credible intervals for our estimates. For a presentation of statistical parameter inference 
with simulation-based inference (SBI) see [Parameter inference with SBI](learning_tutorial_sbi.md).

!!! example

    An example of the following training and inference scripts is presented in
    `tutorials/tutorial_notebooks/07_learning_nn_tutorial.ipynb`.

## Training a NN

In the following, we discuss a supervised learning approach where training data are suitably labeled and the network 
learns to predict the ground truths associated with individual data samples.

The `pypopsyn/learning/train_nn.py` script allows us to train a neural network on a dataset of simulated 
neutron star populations. Once the corresponding heatmaps or 2D arrays have been created (see [Generating density 
maps](generator_tutorial.md) for details), we train the network by running the following command:
```commandline
python pypopsyn/learning/train_nn.py --configuration tutorials/tutorial_notebooks/config_train_cnn.json
```
Here, the `config_train_cnn.json` file contains all the information required to optimize the neural network. If this `CLI` 
argument is left empty, the script will take the default `pypopsyn/learning/config_multiparameter_CNN.json`.

### Configuration options

We now discuss the various options in the `config_train_cnn.json` training configuration file, which include the network
architecture, the input shape of the dataset, or the number of output parameters to predict.

#### General info

We first specify general settings for the experiment such as the experiment's name, the number of GPUs used, the number
of trials performed for each execution of `pypopsyn/learning/train_nn.py` in case convergence is not reached, and the 
specific convergence thresholds for each of the possible output parameters. 
These thresholds represent limiting values for the losses of the individual parameters below which 
we consider the network to have reached sufficient performance in predicting our parameters.

```commandline
{
    "name": "Convolution",
    "trials": 8,
    "convergence": {
        "h_c_threshold": 0.5,
        "vk_c_threshold": 10.0,
        "sigma_k_threshold": 10.0,
        "B_initial_log10_mean_threshold": 0.5,
        "B_initial_log10_sigma_threshold": 0.5,
        "P_initial_log10_mean_threshold": 0.5,
        "P_initial_log10_sigma_threshold": 0.5,
        "a_late_threshold": 0.5,
        "L_radio_log10_mean_threshold": 0.5,
        "epsilon_L": 0.5
    },
    "n_gpu": 0,
}
```

!!! note
    
    Trials and convergence thresholds are relevant because random initialization of network weights and biases can lead 
    to different training behaviour across repeated training runs. To mitigate the possibility of getting trapped in a
    local (but not global) minimum during the optimization, we implement individual thresholds for our parameters and
    perform several training trials until all convergence thresholds are met, or the pre-defined number of trials is 
    reached. In the latter case, the best trained model is saved although not all convergence criteria have been met.


#### Model architecture

Next, we specify the network architecture. In the following example, we use a predefined convolutional neural network 
(CNN) denoted by `ModelConv` defined in `pypopsyn/learning/models/model_conv.py`. This CNN is designed to adapt to 
any input size specified by the `input_shape` parameter and produce an output with a length specified by
`num_parameters`. However, the individual CNN filters and hidden layers are fixed. Below, we specify an input array 
of shape $32 \times 32$ with `3` different input channels (three of our density maps) and an output of `2` values
(corresponding to two of our pulsar population parameters).
```commandline
{
    "arch": {
        "type": "ModelConv",
        "args": {
          "input_shape": [3, 32, 32],
          "num_parameters": 2
        }
    },
}
```
The models that have been predefined are the following ones:

* `ModelConv` based on a CNN.
* `ModelLinear` based on a multi-layer perceptron (MPL) architecture.

If you would like to design your own network architecture, you need to implement a new model class in `pypopsyn/learning/models` and import this model in the file `models.py`.

#### Initialization

We also specify a scheme to initialize the weights and biases of the network. 
Here, we show an example using the Kaiming initializer denoted by `InitializerKaiming`.
```commandline
{
    "weights_initializer": {
      "type": "InitializerKaiming",
      "args": {}
    },
}
```
Here is a list with the different initialization procedures available:

* `InitializerKaiming` uses the Kaiming initialization approach introduced in [He et al. (2015)](https://arxiv.org/abs/1502.01852).
* `InitializerNormal` samples the weights from a Normal distribution $N(0, \sigma^2)$ whilst biases are filled 
 with a constant zero value. In this case, $\sigma = 1 / \sqrt{n}$ where $n$ is the number of input features.
* `InitializerUniform` samples the weights from a uniform distribution $U(0,1)$ whilst biases are filled with 
 a constant zero value.
* `InitializerUniformRule` samples the weights from a uniform distribution $U(-y, y)$. Here, $y = 1 / \sqrt{n}$ 
 with $n$ being the number of input features and biases are filled with a constant zero value.
* `InitializerXavier` uses the Xavier initialization approach introduced in [Xavier et al. (2010)](https://proceedings.mlr.press/v9/glorot10a.html).

#### Training data loader

Next, we define our training data loader, which is responsible for loading the dataset in a representation readable 
by the network. Depending on whether our maps were generated as `.png` images or `.npy` arrays, we set the type to 
`LoaderMultichannelImage` or `LoaderMultichannelArray`, respectively. We also specify the path to the directory 
containing the training dataset (specifically the `dataset_train.csv` file) and the JSON file characterizing the 
statistics of the training dataset, the batch size, the number of workers, i.e., the number of cores used when 
loading the dataset in parallel, the input channels used in building our (multichannel) 
input and the ground truth labels that we want to predict. 
```commandline
{
    "training_data_loader": {
        "type": "LoaderMultichannelArray",
        "args": {
            "dataset_path": "data/example_generator_magrot/dataset_train.csv",
            "statistic_path": "data/example_generator_magrot/statistics_train.json",
            "batch_size": 8,
            "num_workers": 1,
            "filter_inputs": [9, 10, 11],
            "filter_labels": [15, 16],
            "shuffle": true,
            "normalize": false,
            "standardize": false
        }
    },
}
```
The available input channels and labels are specified in the `dataset_train.csv` file, where they are identified with 
an index starting from 0. Let us assume that our `dataset_train.csv` file looks as follows:
```commandline
input:survey_PMPS_position_map_radec,input:survey_SMPS_position_map_radec,input:survey_HTRU_position_map_radec,input:survey_PMPS_velocity_map_vra,input:survey_SMPS_velocity_map_vra,input:survey_HTRU_velocity_map_vra,input:survey_PMPS_velocity_map_vdec,input:survey_SMPS_velocity_map_vdec,input:survey_HTRU_velocity_map_vdec,input:survey_PMPS_ppdot_map,input:survey_SMPS_ppdot_map,input:survey_HTRU_ppdot_map,input:survey_PMPS_ppdot_map_fluxes,input:survey_SMPS_ppdot_map_fluxes,input:survey_HTRU_ppdot_map_fluxes,B_initial_log10_mean,P_initial_log10_mean
data/example_generator_magrot/survey_PMPS_position_map_radec_0.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_0.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_0.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_0.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_0.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_0.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_0.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_0.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_0.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_0.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_0.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_0.npy,12.586280501149703,-1.3096012845994798
data/example_generator_magrot/survey_PMPS_position_map_radec_1.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_1.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_1.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_1.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_1.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_1.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_1.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_1.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_1.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_1.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_1.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_1.npy,12.217080081747753,-0.5284096562094787
data/example_generator_magrot/survey_PMPS_position_map_radec_2.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_2.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_2.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_2.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_2.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_2.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_2.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_2.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_2.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_2.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_2.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_2.npy,13.05641491693826,-0.6189372458073498
data/example_generator_magrot/survey_PMPS_position_map_radec_3.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_3.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_3.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_3.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_3.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_3.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_3.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_3.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_3.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_3.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_3.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_3.npy,12.299076654328934,-1.4327208843495762
data/example_generator_magrot/survey_PMPS_position_map_radec_4.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_4.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_4.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_4.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_4.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_4.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_4.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_4.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_4.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_4.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_4.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_4.npy,13.286135700678276,-1.1544136490794008
```
To select certain input channels, we specify a list containing the indices corresponding to 
the input channels we would like to consider. In the data loader example above, we opt for the input channels 
`survey_PMPS_ppdot_map`, `survey_SMPS_ppdot_map`, `survey_HTRU_ppdot_map` (indices 9, 10, 11) and the labels `B_initial_log10_mean` and `P_initial_log10_mean` (indices 15, 16).

!!! warning
    
    The length of the two lists for `filter_inputs` and `filter_labels` in the dataset loader configuration have to 
    match the channel input dimension and number of output dimension of the neural network.
    Otherwise an error is produced.

Finally, our training data loader enables us to specify if we want to shuffle our training data before passing it
through the neural network and whether we activate on-the-fly normalization or standardization (both are mutually 
exclusive) for the input maps and ground truths (labels). Both take advantage of the statistical information contained 
in the `statistics_train.json` file. For normalization, the input channels and labels will have values in the range between 0 and 1.
If standardized, the input channels and labels have values centred around 0 and range approximately between -1 and 1.

#### Validation data loader

In addition to the training data loader, we also require a loader for the validation set. An example would look like 
this
```commandline
{
    "validation_data_loader": {
        "type": "LoaderMultichannelArray",
        "args": {
            "dataset_path": "data/example_generator_magrot/dataset_train.csv",
            "statistic_path": "data/example_generator_magrot/statistics_train.json",
            "batch_size": 8,
            "num_workers": 1,
            "filter_inputs": [9, 10, 11],
            "filter_labels": [15, 16],
            "shuffle": false,
            "normalize": false,
            "standardize": false
        }
    },
}
```

!!! warning

    Several of the validation loader fields have to be identical to those in the training data loader. In particular,
    the `type`, `filter_inputs`, `filter_labels`, `normalize` and `standardize` options have to be the same, while
    the `batch_size`, `num_workers` and `shuffle` parameters can differ from the training data loader. If the former
    do not match, an error will be raised.

#### Optimizer

We can also set the optimization scheme that regulates the neural network training process (see 
[here](https://pytorch.org/docs/stable/optim.html) for the different types of PyTorch optimizers). Each specific 
optimizer has a set of extra parameters that can be added. For example for the `Adam` optimizer in the example below,
we specify the learning rate `lr` and `weight_decay` rate.
```commandline
{
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 1e-5,
            "weight_decay": 0.0
        }
    },
}
```

#### Loss and accuracy

The `config.json` file also allows us to choose the loss function minimized during the optimization process and the 
metric used to monitor the predictive accuracy of the neural network over the unseen validation set. In the example 
below, the loss function `LossMSE` evaluates the mean square error (MSE) between the output of the network and the 
target labels at every training epoch. To determine the accuracy, a similar metric is implemented through the option 
`MetricAccuracyMSE`. An optimal value of the `MSE` should be around 0 for a well-trained network.
```commandline
{
    "loss": {
      "type": "LossMSE",
      "args": {}
    },
    
    "metric": {
      "type": "MetricAccuracyMSE",
      "args": {}
    },
}
```
Here is a list with the different losses and accuracy metrics available:

* `LossMAE` and `MetricAccuracyMAE` use the mean absolute error (MAE).
* `LossMSE` and `MetricAccuracyMSE` use the mean square error (MSE).
* `LossNLL` uses the negative log-likelihood loss (see [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html) for more details).
* `LossRMSE` and `MetricAccuracyRMSE` use the root mean square error (RMSE).
* `MetricAccuracy` computes the percentage of correctly predicted labels.
* `MetricAccuracyCHI2` uses a reduced $\chi^2$ value.

#### Learning rate scheduler

We can also set up a scheduler for the learning rate. This updates the value of the learning rate after a certain 
number of epochs specified by the `step_size` parameter by multiplying the initial learning rate by a factor `gamma`.

In this example, the learning rate is multiplied by `0.1` after `128` training epochs.
```commandline
{
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
        "step_size": 128,
        "gamma": 0.1
        }
    },
}
```

!!! note

    Scheduling the learning rate has different effects depending on the optimizer used. For example, for an adaptive
    optimizer like `Adam` the learning rate is automatically adjusted during training based on the values of the loss
    gradients with respect to the network weights. Therefore, the learning scheduler will have little effect in this 
    case. On the other hand, for optimizers with fixed learning rates, rescheduling adjustments can help the 
    network converge faster to a minimum of the loss landscape.

#### Training parameters

Finally, we specify some general options for our machine learning experiment. `Epochs` controls the total number of 
epochs we set for our optimization, and `save_dir` the directory path to where we save the best model and relevant 
logging information. The `verbosity` parameter sets the logging level, with `0` for `WARNING`, `1` for `INFO` and `2` 
for `DEBUG`. Moreover, the `save_period` parameter sets the number of epochs after which the network status is 
saved during training even if convergence has not been reached yet. We can also set an `early_stop`, i.e., the 
number of epochs after which stop training if the validation loss is not improving, in order to avoid overfitting.
Finally, the `tensorboard` option controls if the training info will be saved into a tensorboard log file.
```commandline
{
    "trainer": {
        "epochs": 1024,
        "save_dir": "data/example_learning_nn",
        "save_period": 1000,
        "verbosity": 1,
        "early_stop": 32,
        "tensorboard": true
    }
}
```

### Varying parameters via `CLI`

Instead of passing a `config.json` file to the training module, we can also provide some (but not all) of the 
parameters in the configuration file directly via `CLI` when launching the training script.

In particular, we can provide the paths to the training and validation datasets, the statistics JSON file, 
the input channels and the labels to select, the input shape, the number of parameters to predict, the option to 
apply normalization or standardization (where `0` equals `false` and `1` equals `true`), the batch size, the learning 
rate value and the path to where the trained model is saved.

For example, we can train a neural network as follows:
```commandline
python pypopsyn/learning/train_nn.py --configuration tutorials/tutorial_notebooks/config_train_cnn.json --dataset_training data/example_generator_magrot/dataset_train.csv --dataset_validation data/example_generator_magrot/dataset_test.csv --dataset_statistics data/example_generator_magrot/statistics_train.json --filter_inputs 9 10 11 --filter_labels 15 16 --input_shape 3 32 32 --num_parameters 2 --batch_size 1 --lr 1e-5 --save_dir learning_results
```

### Training output

No matter which of the two ways are used to launch a training experiment, the results of the training experiment are 
saved in the directory specified by the `save_dir` option (either in the configuration file or via `CLI`). Specifically, 
training will create two folders in this directory, namely a `logs` folder and a `models` folder. Both contain 
subfolders for each specific training experiment of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes 
the date and time when the experiment was launched.

Each subfolder in the `logs` directory contains a `log.txt` file with the terminal output and the following three `.json` files:

* `train_results.json` with the training loss evolution as a function of training epochs. 
    If the labels where normalized or standardized, this training loss is also normalized or standardized
    and will not have physical units.
* `train_eval_results.json` with the training loss evolution in physical units, i.e., we have corrected for
     normalization or standardization if they were applied.
* `valid_results.json` with the validation loss evolution in physical units, i.e., we have corrected for
     normalization or standardization if they were applied.

In these files, for each training epoch, we report the loss for each parameter that our network learns plus the total 
loss, simply named `loss`, which is the sum of the losses of all trainable parameters. In addition, these files also 
quote an accuracy metric for each epoch, which in the case of `MetricAccuracyMAE`, `MetricAccuracyMSE` and 
`MetricAccuracyRMSE` is the average of the losses of the individual parameters.

Finally, each subfolder in the `models` directory contains the saved best-trained model in `.pth` format. This model
will be used when inferring on an unseen dataset as outlined below.

## Inferring on a dataset

Once a network has been trained, it can be used to infer on an unseen dataset of generated maps and extract the 
relevant pulsar population parameters. The script `pypopsyn/learning/infer_nn.py` allows us to take an experiment
configuration file, a pre-trained model, and a dataset, and run inference on selected samples from that dataset.

### Configuration options

We again set up the loader for the test dataset over which we want to perform inference in the configuration file.
As noted above for the validation data loader the test data loader must have the same `type`, `filter_inputs` and 
`filter_labels` as the training data loader. Again, both have to be compatible with the network's input and output 
shape. Moreover, the test dataset has to be normalized or standardized if normalization or standardization was 
applied to the training dataset.

The test data loader in the `config.json` file could look as follows:
```commandline
{
    "test_data_loader": {
        "type": "LoaderMultichannelArray",
        "args": {
            "dataset_path": "data/example_generator_magrot/dataset_test.csv",
            "statistic_path": "data/example_generator_magrot/statistics_train.json",
            "batch_size": 1,
            "num_workers": 1,
            "filter_inputs": [9, 10, 11],
            "filter_labels": [15, 16],
            "shuffle": false,
            "normalize": false,
            "standardize": false
        }
    },
}
```
Note that we set the `batch_size` to `1` here as we want to infer on individual samples not batches. As for the 
validation data, we do not require our samples to be shuffled.

We specify the directory for saving the inference results in the field `infer`.
```commandline
{
    "infer": {
        "save_dir": "data/example_inference_nn"
    }
 }
```

### Inference output

To use the inference script, we need to provide a pretrained model (`--trained_model`) and the configuration file of 
the experiment that generated the corresponding model (`--configuration`). For instance, we could run the following
command:
```commandline
python pypopsyn/learning/infer_nn.py --configuration tutorials/tutorial_notebooks/config_train_cnn.json --trained_model data/example_learning_nn/models/Convolution/20240926_113426/best_model_trial1.pth
```

We can also use the `--samples` argument to provide a list of specific samples we would like to infer on (their indices 
in the dataset). Leaving this option blank implies that we are inferring over all test data samples. As for the 
training script, we can also specify several other parameters directly via `CLI`. To see relevant options run
```commandline
python pypopsyn/learning/infer_nn.py --help
```

As noted above, the dataset used for inference has to have the same configuration as the training dataset used during
the network optimization process. If the inference dataset does not match, PyTorch automatically raises an error.
For example, using the wrong resolution for the input maps raises an error similar to:
```commandline
RuntimeError: Error(s) in loading state_dict for ModelConv:
size mismatch for fc1.weight: copying a param with shape torch.Size([64, 26880]) from checkpoint, the shape in current model is torch.Size([64, 12544]).
```
If the input channels do not match, an error like the following is raised:
```commandline
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 128 and the array at index 1 has size 64
```

If the inference is successful, the output `pypopsyn/learning/infer_nn.py` will be saved in the directory specified in the 
`save_dir` option. Specifically, inference will create a `logs` folder in this directory containing subfolders of the 
form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes the date and time when the inference was launched.

Each subfolder in the `logs` directory will contain an `inference_results.csv` file containing the labels (ground 
truths) for each sample and the corresponding predictions from the trained model. For example, in case of inference for 
the two parameters `P_initial_log10_mean` and `B_initial_log10_mean` for four test samples, the output file could 
look like this:
```commandline
target:B_initial_log10_mean,target:P_initial_log10_mean,predicted:B_initial_log10_mean,predicted:P_initial_log10_mean
12.187267303466797,-1.3277602195739746,12.33946418762207,-1.3803331851959229
12.238884925842285,-1.4647713899612427,12.271954536437988,-1.4287304878234863
12.566689491271973,-0.331459105014801,12.784342765808105,-0.3929998278617859
13.581913948059082,-1.2630716562271118,13.486227989196777,-1.055734395980835
```
