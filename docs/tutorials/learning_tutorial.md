# Training a CNN for pulsar population synthesis

We use here a supervised learning approach where training data are suitably labeled and the network has to learn to predict the target value of the label associated to the data samples.
The `pypopsyn/learning/train.py` script allows to train a neural network over a dataset of samples of simulated neutron star populations.
Once the dataset containing the heatmaps or 2D arrays has been created, to train the network over the dataset one can run the script:
```commandline
python pypopsyn/learning/train.py --configuration config.json
```
where the `config.json` file contains all the information needed by the network to train. This `CLI` can be left unspecified and the script will take the default `pypopsyn/learning/config_multiparameter_MLP.json`.

In this file we can specify various options to configure the training process, e.g., the network model architecture to use, the input shape of the dataset, or the number of output parameters to predict.

First of all, we can specify some general settings such as the name of the experiment, the amount of GPUs needed for it, the amount of trials to perform if convergence is not reached, and the specific thresholds for convergence for each one of the predicted or output parameters.
The script will try to perform several training trials until all convergence thresholds are met or the number of trials is reached.
If convergence is not reached in the number of trials indicated the best trained model is saved anyway.
```commandline
{
    "name": "Linear",
    "trials": 8,
    "convergence": {
        "h_c_threshold": 0.5,
        "vk_c_threshold": 10.0,
        "sigma_k_threshold": 10.0,
        "B_initial_log10_mean_threshold": 0.5,
        "B_initial_log10_sigma_threshold": 0.5,
        "P_initial_log10_mean_threshold": 0.5,
        "P_initial_log10_sigma_threshold": 0.5,
        "a_late_threshold": 0.5
    },
    "n_gpu": 0,
}
```

The first section we need to specify is the architecture. 
For the sake of the example, we are using a convolutional neural network (CNN) which is receiving an array with shape $32 \times 32$ with `3` different input channels and is giving as output the predicted value of one parameter for each sample:
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

We also need to specify a scheme to initialize the weights and biases of the network. 
In this case, we show an example using the `InitializerKaiming`.
```commandline
{
    "weights_initializer": {
      "type": "InitializerKaiming",
      "args": {}
    },
}
```

Next, we need a training loader, responsible for loading the dataset for the training in a representation readable by the network.
Here you have to specify the path to the folder containing the training dataset, the batch size, and the eventual input channels to use in building a multichannel input.
Additionally, we can choose the list of labels from the dataset that we want to consider.
The available input channels and labels are specified in the `train_dataset.csv` file and here they are identified with an index starting from 0.
To select some input channels you need to specify a list containing the indices corresponding to the input channels you would like to consider.
In the example below we are selecting the input channels `survey_PMPS_ppdot_map`, `survey_SMPS_ppdot_map`, `survey_HTRU_ppdot_map` and the labels `B_initial_log10_mean` and `P_initial_log10_mean`.
Furthermore, we can enable on-the-fly normalization or standardization (mutually excluding) for both inputs and labels.
This will use the statistical information contained in the `statistics_train.json` file.
If normalized the input channels will have values in the range between 0 and 1.
If standardized the input channels have values centred around 0 and ranging approximately between -1 and 1.
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
            "filter_labels": [12, 13],
            "shuffle": true,
            "normalize": false,
            "standardize": false
        }
    },
}
```

We need to provide a loader for the validation set using the `validation_data_loader`.
Such loader must have the same `filter_inputs` and `filter_labels` and be of the same `type` as the training data loader.
In fact, what matters is that both of them are compatible with the network's input shape.
Moreover, the test dataset has to be normalized or standardized if normalization or standardization was applied to the training dataset.
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
            "filter_labels": [12, 13],
            "shuffle": false,
            "normalize": false,
            "standardize": false
        }
    },
}
```

We can set the optimizer type, which regulates the training process (see `here <https://pytorch.org/docs/stable/optim.html>`_ for the different type of PyTorch optimizers). 
Each specific optimizer has a set of extra parameters that can be provided (such as `lr` or `weight_decay` for ADAM).
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

We have to specify the loss function to minimize and the metric to monitor the predictive accuracy of the neural network model over the validation set during training. 
The implemented loss function `LossMSE` evaluates the mean square error (MSE) between the output of the network and the target labels over every training epoch. 
For the accuracy metric a similiar `MSE` metric is implemented called `MetriAccuracyMSE`. An optimal value of the `MSE` should be around 0 for a well-trained network:
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

We can also set up a scheduler for the learning rate, which can update the value of the learning rate after a number of epochs specified by the `step_size` parameter, by multiplying it by a factor `gamma`.
In this example after `128` training epochs the learning rate is multiplied by a factor `0.1`.
Note that scheduling has different effects depending on the optimizer.
For example for an adaptive optimizer like ADAM the learning rate is automatically adjusted during training, depending on the values of the loss gradients with respect to the network weights.
Therefore the learning scheduler could not be effective in this case.
On the other hand, for optimizer where the learning rate is fixed, rescheduling its value after some training epochs could help to converge faster towards a minimum of the loss landscape.
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

Some parameters for the training routine such as the total number of epochs, the directory path where to save the best model network and some checkpoints during the training process.
```commandline
{
    "trainer": {
        "epochs": 1024,
        "save_dir": "data/example_training",
        "save_period": 1000,
        "verbosity": 1,
        "early_stop": 32,
        "tensorboard": true
    }
}
```

Once you have set up the configuration file, to launch the training script you can simply run the script:
```commandline
python pypopsyn/learning/train.py --configuration config.json
```

When launching the training script you can also provide some of the parameters contained in the configuration file directly via `CLI`.
For example one can provide the paths to the training and validation dataset, the input channels and the labels to select, the input shape, the number of parameters to predict, either to apply normalization or standardization to the input, the batch size, the learning rate value and the path where to save the trained model.
For example you can run a script like the following:
```commandline
python pypopsyn/learning/train.py --configuration config.json --dataset_training generated_dataset/dataset_train.csv --dataset_validation generated_dataset/dataset_valid.csv --dataset_statistics generated_dataset/statistics_train.json --filter_inputs 0 3 4 5 --filter_labels 14 --input_shape 4 64 64 --num_parameters 1 --normalize 1 --batch_size 1 --lr 1e-5 --save_dir training_results
```

The results of the training will be saved in the directory specified under the key `["trainer"]["save_dir"]` specified in the configuration file.
In this directory path two folders will be created, a `logs` folder and a `models` folder that will contain subfolders for each specific training experiment with the structure of the form `name/YYYYMMDD_HHMMSS` where `YYYYMMDD_HHMMSS` denotes the date and time when the experiment was performed with a particular `name` specified in the configuration file.
Each subfolder in the `logs` directory will contain three `json` files:
* `train_results.json` containing the training loss evolution. if the labels where normalized or standardized the training loss here will be also normalized or standardized i.e. it will not have physical units.
* `train_eval_results.json` containing the training loss evolution in physical units (i.e. with normalization or standardization removed if they where applied).
* `valid_results.json` containing the validation loss evolution in physical units (i.e. with normalization or standardization removed if they where applied).
Each subfolder in the `models` directory will contain the saved best trained model in `.pth` format.


## Infer on a Data Set

Once a network has been trained, it can be used to infer on an existing dataset of generated maps.
To do so the script `pypopsyn/learning/infer.py` allows you to take an experiment configuration file, a pretrained model, and a data set to run inference on selected samples for that dataset.

You can setup the loader for the test dataset over which you would like to perform inference in the configuration file.
Such loader must have the same `filter_inputs` and `filter_labels` and be of the same `type` as the training data loader.
In fact, what matters is that both of them are compatible with the network's input shape.
Moreover, the test dataset has to be normalized or standardized if normalization or standardization was applied to the training dataset.
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
            "filter_labels": [12, 13],
            "shuffle": false,
            "normalize": false,
            "standardize": false
        }
    },
}
```

The directory where to save the inference results is specified in the field `infer`.
```commandline
{
    "infer": {
        "save_dir": "data/example_inference"
    }
 }
```

To use this inference script you will need to provide a checkpoint with a pretrained model (`--trained_model`) and the configuration file of the experiment that generated such model (`--configuration`). For instance:
```commandline
python pypopsyn/learning/infer.py --configuration config.json --trained_model output/learning/models/Convolution/20240725_175854/best_model_trial1.pth
```

You can use the `--samples` argument to provide a list of samples you would like to infer (their indices in the dataset) or just leave it blank to infer over all.
Make sure that the data set used for inference has the same input configuration as the data set used for training the model, i.e., same input shape, number of labels to predict, normalization etc..
If the inference dataset does not match an error is raised automatically by pytorch.
For example if you put the wrong resolution for the input maps the error raised is similar to:
```commandline
RuntimeError: Error(s) in loading state_dict for ModelConv:
size mismatch for fc1.weight: copying a param with shape torch.Size([64, 26880]) from checkpoint, the shape in current model is torch.Size([64, 12544]).
```
If the input channels do not match, an error like the following is raised:
```commandline
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 128 and the array at index 1 has size 64
```
The output of the inference script will be saved in the file `logs/inference_results.csv` in the path specified by the `save_dir` in the configuration file.
This file will contain the labels (ground truth) for each sample and the corresponding predictions from the trained model.
For example, in case of inference over the two parameters `P_initial_log10_mean` and `B_initial_log10_mean` for four test samples the output file would be like this:
```commandline
target:B_initial_log10_mean,target:P_initial_log10_mean,predicted:B_initial_log10_mean,predicted:P_initial_log10_mean
12.187267303466797,-1.3277602195739746,12.33946418762207,-1.3803331851959229
12.238884925842285,-1.4647713899612427,12.271954536437988,-1.4287304878234863
12.566689491271973,-0.331459105014801,12.784342765808105,-0.3929998278617859
13.581913948059082,-1.2630716562271118,13.486227989196777,-1.055734395980835
```

To see a tutorial example for how to use this training and inference scripts you can look at the notebook in `tutorials/tutorial_notebooks/learning_tutorial.ipynb`.
