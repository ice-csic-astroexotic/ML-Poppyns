# Parameter inference with SBI

In the following, we describe several scenarios of simulation-based inference (SBI) with neural networks. This is a 
supervised learning approach where a network is trained to learn the mapping between simulated data and the posterior 
distribution of the underlying model parameters. 

The framework thus allows us to perform robust statistical inference and derive credibility intervals for parameter 
estimation with complex simulators like those developed for pulsar population synthesis. Our implementation builds on 
the [sbi](https://sbi-dev.github.io/sbi/) library ([Tejero-Cantero et al., 2020](https://arxiv.org/abs/2007.09114)).

For a discussion of how neural networks can be used to infer point estimates (without quantifying uncertainties) see
[Learning pulsar parameters with NNs](learning_tutorial_cnn.md).

## Amortized NPE

The `pypopsyn/learning/train_sbi.py` script employs an SBI method called Neural Posterior Estimation (NPE) that allows 
us to train a neural density estimator on a dataset of samples of simulated neutron star populations to directly 
approximate the posterior distributions of the input parameters. In this case, the inference is amortized, which 
implies that the trained model is able to predict a posterior distribution for any synthetic population of neutron 
stars. This computation is very fast, because inference corresponds to a simple forward pass through the network.

!!! example

    An example of the following training and inference scripts is presented in
    `tutorials/tutorial_notebooks/08_learning_sbi_tutorial.ipynb`.

To perform our NPE using a dataset composed of heatmaps or 2D arrays of our synthetic pulsar populations (see 
[Generating density maps](generator_tutorial.md) for details) we use the script `pypopsyn/learning/train_sbi.py` as 
follows:
```commandline
python pypopsyn/learning/train_sbi.py --configuration tutorials/tutorial_notebooks/config_sbi.json
```
Here, the `config_sbi.json` file contains all the information required to optimize the neural network. If this `CLI` 
argument is left empty, the script will take the default `pypopsyn/learning/config_sbi.json`.

### Configuration options

We now discuss the various options in the `config_sbi.json` training configuration file, which include the network
architecture for our embedding net (see below), the type of density estimator, the input shape of the dataset, and 
other relevant training hyperparameters.

#### General info

We first specify general settings for the experiment such as the experiment's name, the number of GPUs used, 
a manual seed for the initialization of the network weights (assuming that `set_manual_seed` is set to true) 
and some additional profiling options.
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

#### Model architecture

Next, we specify the architecture for the so-called embedding neural network. This embedding net is used to extract
features from the input data and compress the input into a latent vector that is then passed to the density estimator.

In the following example, we will be using 2D maps as input and, hence, opt for a convolutional neural network (CNN) 
as the embedding net. This CNN is designed to adapt to any input size specified by the `input_shape` parameter and produce an output with a length specified by `len_output_layer`.
In this specific example the CNN receives an array of shape $32 \times 32$ with `3` different input channels 
(three of our density maps with a 32 resolution) as input and outputs a latent vector of size 32, which contains 
a compressed representation of the input feature maps.

The configuration file then looks as follows:
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
If you would like to design your own model architecture for the embedding network you need to implement a new model class in `pypopsyn/learning/models` and import it in the file `models.py`.

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
Here is a list with the different initialization procedure you could adopt:

* `InitializerKaiming` uses the Kaiming initialization approach introduced in [He et al. (2015)](https://arxiv.org/abs/1502.01852).
* `InitializerNormal` samples the weights from a Normal distribution $N(0, \sigma^2)$ whilst biases are just filled with a constant zero value. In this case, $\sigma = 1 / \sqrt{n}$ where $n$ is the number of input features.
* `InitializerUniform` samples the weights from a uniform distribution $U(0,1)$ whilst biases are just filled with a constant zero value.
* `InitializerUniformRule` samples the weights from a uniform distribution $U(-y, y)$ where $y = 1 / \sqrt{n}$ being $n$ the number of input features and biases are just filled with a constant zero value.
* `InitializerXavier` uses the Xavier initialization approach introduced in [Xavier et al. (2010)](https://proceedings.mlr.press/v9/glorot10a.html).

#### Density estimator

Next, we decide on the type of density estimator used to approximate the posterior distribution. For NPE, the 
preconfigured options in the sbi library include so-called masked autoregressive flows `maf` or Gaussian mixture 
density networks `mdn`. For more details on these methods and relevant hyperparameters as well as custom density 
estimators see [here](https://sbi-dev.github.io/sbi/latest/tutorial/04_density_estimators/).

In the following example, we are setting a mixture density network with `10` Gaussian components.
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

#### Training data loader

We also require a training data loader, which is responsible for loading the dataset in a representation readable 
by the network. We specify the path to the directory containing the training dataset (specifically the 
`dataset_train.csv` file) and the JSON file characterizing the statistics of the training dataset, and the input 
channels used in building our (multichannel) input. Additionally, we set the ground truth labels that we want to 
predict.

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
`survey_PMPS_ppdot_map`, `survey_SMPS_ppdot_map`, `survey_HTRU_ppdot_map` (indices 9, 10, 11) and the labels 
`B_initial_log10_mean` and `P_initial_log10_mean` (indices 15, 16).

!!! warning
    
    The length of the two lists for `filter_inputs` and `filter_labels` in the dataset loader configuration have to 
    match the channel input dimension and number of output dimension of the neural network.
    Otherwise an error is produced.

Finally, our training data loader enables us to activate on-the-fly normalization or standardization (both are mutually 
exclusive) for the input maps and ground truths (labels). Both take advantage of the statistical information contained 
in the `statistics_train.json` file. For normalization, the input channels and labels will have values in the range between 0 and 1.
If standardized, the input channels and labels have values centred around 0 and range approximately between -1 and 1.

!!! note

    The training dataset is also used for validation. This is automatically handled through sbi. We therefore do
    not require a separate validation data loader. We can specify the corresponding validation fraction as outlined 
    below.

#### Training parameters

Finally, we specify some general options for our machine learning experiment. In particular, we can set the fraction 
of the training dataset that we want to use for validation, the training batch size, the initial learning rate for the
`Adam` optimizer (the default in the sbi library) and the directory path where the trained model is saved.
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

### Varying parameters via `CLI`

Instead of passing a `config.json` file to the training module, we can also provide some (but not all) of the 
parameters in the configuration file directly via `CLI` when launching the training script.

In particular, we can provide the path to the training datasets and the statistics JSON file, 
the input channels and the labels to select, the input shape, the length of the latent vector, the option to 
apply normalization or standardization (where `0` equals `false` and `1` equals `true`), the batch size, the learning 
rate value and the path to where the trained model is saved.

For example, we can train a neural network as follows:
```commandline
python pypopsyn/learning/train_sbi.py --configuration tutorials/tutorial_notebooks/config_sbi.json --dataset_training data/example_generator_magrot/dataset_train.csv --dataset_statistics data/example_generator_magrot/statistics_train.json --filter_inputs 9 10 11 --filter_labels 15 16 --input_shape 3 32 32 --len_output_layer 32 --standardize 1 --batch_size 1 --lr 1e-5 --save_dir learning_sbi_results
```

### Training output

No matter which of the two ways are used to launch a training experiment, the results of the training experiment are 
saved in the directory specified by the `save_dir` option (either in the configuration file or via `CLI`). Specifically, 
training will create two folders in this directory, namely a `logs` folder and a `models` folder. Both contain 
subfolders for each specific training experiment of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes 
the date and time when the experiment was launched.

Moreover, each subfolder in the `logs` directory contains the following files:

* `profile.json` and `profile.log` files with profiling information of the training experiment.
* `training_statistics.json` with the training and validation loss evolution.
* `training_stats.pdf` with a plot showing the training and validation loss evolution.

Finally, each subfolder in the `models` directory contains the saved best trained model in `.pickle` format. This 
model will be used for the inferring on an unseen dataset as outlined below.

## Inferring on a dataset

Once our SBI pipeline (composed of an embedding net and a neural density estimator) has been trained, it can be used 
to infer on an unseen dataset of generated maps and extract posterior distributions of the corresponding pulsar
population parameters. The script `pypopsyn/learning/infer_sbi.py` allows us to take an experiment configuration file, 
a pre-trained model, and a dataset, and run the inference.

### Configuration options

In addition to the training data loader specified above, we also require a loader for the test set to perform 
inference. We again need to set the path for the test dataset and the corresponding statistics (note that we use the 
same file as for training here to make sure we capture the full range of training dataset properties), the input 
channels, labels and preprocessing steps. An example would look like this:
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

!!! warning

    The test data loader requires the same set-up as the training data loader. That means that the `filter_inputs`,
    `filter_labels`, `normalize` and `standardize` options have to be indentical to those in the training data loader.

We also specify the directory for saving the inference results in the field `infer`.
```commandline
{
    "infer": {
        "save_dir": "data/example_inference_sbi"
    }
}
```

### Inference output

Once the inference configuration is set up, we run the inference script by providing the configuration file 
(`--configuration`) and a pretrained model (`--trained_model`) as follows:
```commandline
python pypopsyn/learning/infer_sbi.py --configuration config_sbi.json --trained_model data/example_training_sbi/models/SBI_ConvolutionMDN/20240626_105721/trained_model.pickle
```

As for the training script, we can also specify several other parameters, such as the path to the test dataset,
the input channels and the labels directly via `CLI`. To see all available options run
```commandline
python pypopsyn/learning/infer.py --help
```
One additional option that is particularly useful for assessing the quality of the inference is setting the 
argument `--corner_plot` to `True`. This produces and saves a corner plot of the two- and one-dimensional 
marginalised posteriors for each of the test samples in `.pdf` format and the corresponding posterior samples in 
`.pt` format.

!!! warning

    Producing corner plots for a large test dataset is computationally expensive due to the computational cost
    of sampling from the posterior and should be avoided.


If the inference is successful, the output of `pypopsyn/learning/infer_sbi.py` will be saved in the directory 
specified in the `save_dir` option. Specifically, inference will create a `logs` folder in this directory containing
subfolders of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes the date and time when the inference
was launched.

Each subfolder in the `logs` directory contains the following information:

* `profile.json` and `profile.log` files with the timing profiling of the inference script.
* `coeff_gaussians.csv` file with the coefficients for the Gaussian mixture components for each test sample.
* `coverage_plot.pdf` and `coverage_probability.npy` files with the results of the coverage probability diagnostic test.
* the corner plots in `.pdf` format for each of the test samples if the option `--corner_plot` is set to `True`.


## Infer on a Data Set with an Ensemble


## Truncated Sequential Neural Posterior Estimation


  

  
  
