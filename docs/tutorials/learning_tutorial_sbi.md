# Parameter inference with SBI

In the following, we describe several scenarios of simulation-based inference (SBI) with neural networks. This 
framework allows us to perform robust statistical inference and derive credibility intervals for parameter estimation 
with complex simulators like those developed for pulsar population synthesis. Our implementation builds on the 
[sbi](https://sbi-dev.github.io/sbi/) library ([Tejero-Cantero et al., 2020](https://arxiv.org/abs/2007.09114)). 
For a discussion of how neural networks can be used to infer point estimates (without quantifying uncertainties) see
[Learning pulsar parameters with NNs](learning_tutorial.md).

## Neural Posterior Estimation (amortized)

We use here a simulation-based inference framework that uses the library [sbi](https://sbi-dev.github.io/sbi/ ).
This is a supervised learning approach where a network is trained to learn the mapping between simulated data and the posterior distribution of the model parameters used to simulate them.

The `pypopsyn/learning/train_sbi.py` script employs a sbi method called Neural Posterior Estimation (NPE) that allows to train a neural density estimator over a dataset of samples of simulated neutron star populations to directly approximate the posterior distributions of the input parameters.
In this case the inference is amortized, meaning that the trained model is able to predict a posterior distribution for any input simulated population of neutron stars.

Once the dataset containing the heatmaps or 2D arrays has been created, to train the network over the dataset one can run the script:
```commandline
python pypopsyn/learning/train_sbi.py --configuration config_sbi.json
```

where the `config_sbi.json` file contains all the information needed by the network to train.
This `CLI` can be left unspecified and the script will take the default `pypopsyn/learning/config_sbi.json`.

In this file we can specify various options to configure the training process, e.g., the network model architecture to use as an embedding net, the density estimator type, the input shape of the dataset, and some training hyperparameters.

First of all, we can specify some general settings such as the name of the experiment, the amount of :term:`GPU`\s needed for it, a manual seed for the initialization of the network weights and some profiling options.

```commandline
{
    "name": "SBI_ConvolutionMDN",
    "n_gpu": 1,
    "set_manual_seed": false,
    "manual_seed": 42,
    "profile_log": "profile.log",
    "profile_json": "profile.json",
    "show_profiling": true,
}
```

The first section we need to specify is the architecture of the embedding neural network.
The embedding network is used to extract features from the input data and to compress them into a latent vector that will be passed to the density estimator.
Since in this example we are using 2D maps as input for the sake of the example, we are using a convolutional neural network (CNN) which is receiving an array with shape $32 \times 32$ with 3 different input channels and is giving a latent vector of size 32 containing a compressed representation of the input feature maps:
```commandline
{
    "arch": {
        "type": "ModelConvSBI",
        "args": {
            "input_shape": [3, 32, 32],
            "len_output_layer": 32
        }
    },
}
```

We also need to specify a scheme to initialize the weights and biases of the network. In this case, we show an example using the `InitializerKaiming`.
```commandline
{
    "weights_initializer": {
        "type": "InitializerKaiming",
        "args": {}
    },
}
```

We can also specify the type of density estimator used to approximate the posterior distribution.
In this example we are setting a mixture density network (mdn) with 10 Gaussian components.
```commandline
{
    "density_estimator": {
        "type": "mdn",
        "args": {
            "num_components": 10
        }
    },
}
```

Next, we need a training loader, responsible for loading the dataset for the training in a representation readable by the network.
Here you have to specify the path to the folder containing the training dataset, the eventual input channels to use in building a multichannel input.
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
        "dataset_path": "data/example_generator_magrot/dataset_train.csv",
        "statistic_path": "data/example_generator_magrot/statistics_train.json",
        "filter_inputs": [9, 10, 11],
        "filter_labels": [15, 16],
        "normalize": false,
        "standardize": true
    },
}
```

We need to provide a loader for the test set using the `test_data_loader`.
Such loader must have the same `filter_inputs` and `filter_labels` and the same `normalize` or `standardize`.
In fact, what matters is that both of them are compatible with the network's input shape.
```commandline
{
    "test_data_loader": {
        "dataset_path": "data/example_generator_magrot/dataset_test.csv",
        "statistic_path": "data/example_generator_magrot/statistics_train.json",
        "filter_inputs": [9, 10, 11],
        "filter_labels": [15, 16],
        "normalize": false,
        "standardize": true
    },

}
```

We can set some hyperparameters related to the trainer, i.e. the fraction of the training dataset to use for validation, the training batch size, the initial learning rate for the `Adam` optimizer and the directory path where to save the trained model.
```commandline
{
    "trainer": {
        "validation_fraction": 0.1,
        "batch_size": 8,
        "lr": 5e-4,
        "save_dir": "data/example_learning_sbi"
    },
}
```

Finally, we can set up the directory path where the inference results on the test set will be saved.
This is not used during the training process but will be necessary when doing inference (see next section).
```commandline
{
    "infer": {
        "save_dir": "data/example_inference_sbi"
    }
}
```

Once you have set up the configuration file, to launch the training script you can simply run the script:
```commandline
python pypopsyn/learning/train_sbi.py --configuration config_sbi.json
```

When launching the training script you can also provide some of the parameters contained in the configuration file directly via `CLI`.
For example one can provide the paths to the training, the input channels and the labels to select, the input shape, either to apply normalization or standardization to the input, the batch size, the learning rate value and the path where to save the trained model.
For example, you can run a script like the following:
```commandline
python pypopsyn/learning/train_sbi.py --configuration config_sbi.json --dataset_training data/example_generator_magrot/dataset_train.csv --dataset_statistics data/example_generator_magrot/statistics_train.json --filter_inputs 9 10 11 --filter_labels 12 13 --input_shape 3 32 32 --len_output_layer 32 --standardize 1 --batch_size 1 --lr 1e-5 --save_dir data/example_learning_sbi
```

The results of the training will be saved in the directory specified under the key `["trainer"]["save_dir"]` in the configuration file.
In this directory path two folders will be created, a `logs` folder and a `models` folder that will contain subfolders for each specific training experiment with the structure of the form `name/YYYYMMDD_HHMMSS` where `YYYYMMDD_HHMMSS` denotes the date and time when the experiment was performed with a particular `name` specified in the configuration file.
Each subfolder in the `logs` directory will contain the following files:
* `profile.json` and `profile.log` files containing the timing profiling of the training script.
* `training_statistics.json` containing the training and validation loss evolution.
* `training_stats.pdf` containing a plot showing the training and validation loss evolution.
Each subfolder in the `models` directory will contain the saved best trained model in `.pickle` format.


## Infer on a Data Set

Once a network has been trained, it can be used to infer on an existing dataset of maps.
To do so the script `pypopsyn/learning/infer_sbi.py` allows you to take an experiment configuration file, a pretrained model, and a data set to run inference.

Once the configuration file is properly set up with the path to the test dataset, to run the inference script you can use a command similar to the following one:
```commandline
python pypopsyn/learning/infer_sbi.py --configuration config_sbi.json --trained_model data/learning_sbi/models/SBI_ConvolutionMDN/20240606_180938/trained_model.pickle
```

As for the training script you could provide some arguments via `CLI`, for example the path to the test dataset, the input channels and the labels to select, the input shape, either to apply normalization or standardization to the input and so on.

The output of the inference script will be saved in the directory specified under the key `["infer"]["save_dir"]` in the configuration file.
In this directory path a folder `logs` will be created that will contain subfolders for each specific training experiment with the structure of the form `name/YYYYMMDD_HHMMSS` where `YYYYMMDD_HHMMSS` denotes the date and time when the experiment was performed with a particular `name` specified in the configuration file.
Each subfolder in the `logs` directory will contain the following information:
* `profile.json` and `profile.log` files containing the timing profiling of the inference script.
* `coeff_gaussians.csv` file will contain the Gaussian coefficients for the components of the Gaussian mixture for each of the test sample.
* `coverage_plot.pdf` and `coverage_probability.npy` files will contain the results of the coverage probability diagnostic test.
You could also specify the argument `--corner_plot True` while launching the script in order to produce and save the posterior corner plots in `.pdf` format and the posterior samples in `.pt` format for each of the test samples.

!!! example

    To see a tutorial example for how to use these training and inference scripts you can look at the notebook in `tutorials/tutorial_notebooks/08_learning_sbi_tutorial.ipynb`.


## Infer on a Data Set with an Ensemble


## Truncated Sequential Neural Posterior Estimation


  

  
  
