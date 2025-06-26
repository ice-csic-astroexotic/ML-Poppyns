# Parameter inference with SBI

In the following, we describe several scenarios of simulation-based inference (SBI) with neural networks. This is a 
supervised learning approach where a network is trained to learn the mapping between simulated data and the posterior 
distribution of the underlying model parameters. 

The framework thus allows us to perform robust statistical inference and derive credibility intervals for parameter 
estimation with complex simulators like those developed for pulsar population synthesis. Our implementation builds on 
the [sbi](https://sbi-dev.github.io/sbi/) library ([Tejero-Cantero et al., 2020](https://arxiv.org/abs/2007.09114)).

For a discussion of how neural networks can be used to infer point estimates (without quantifying uncertainties) see
[Learning pulsar parameters with NNs](learning_tutorial_nn.md).

## SBI methods

Using the `pypopsyn/learning/sbi_train.py` script, we support the following approaches for simulation-based inference:

 1. Neural Posterior Estimation ([NPE](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)): A neural network is trained to approximate the posterior distribution directly, 
    learning a mapping from model parameters **θ** to **P(θ | x)**.

 2. Neural Likelihood Estimation ([NLE](https://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)): A neural network emulates the simulator by approximating the likelihood 
    **P(x | θ)**. Once trained, this model can be used with standard sampling algorithms (e.g., MCMC) to sample from 
    the posterior.

 3. Neural Ratio Estimation ([NRE](https://proceedings.mlr.press/v119/hermans20a/hermans20a.pdf)): A classifier is trained to approximate the likelihood-to-evidence ratio
    **r(θ, x) = P(x | θ) / P(x)**, which can be used to compute or sample from the posterior using methods like MCMC.

## Amortized vs Sequential inference

SBI can be performed using either amortized or sequential strategies. While amortized inference enables fast posterior 
estimates for any input after a single large training phase, sequential methods focus the simulation budget on regions 
most relevant to a specific observation, making them more efficient for cases with a single dataset and expensive 
simulations, as in our application. In the following, we will refer to amortized as **single-round** and sequential 
as **multi-round** inference. All the methods above have their own sequential variants, which are named by adding an 'S'
at the beginning. For example, SNPE stands for Sequential Neural Posterior Estimation. 

For multi-round inference, the workflow is as follows:

1. Sample the proposal prior distribution to obtain $\theta_i \sim P(\theta)$.
2. Given the parameter samples from step 1, generate synthetic data $x \sim P(x|\theta_i)$ using the simulator.
3. Train the neural network on the training dataset consisting of pairs $(\theta_i, x_i)$ obtained in the previous steps.
4. Use the trained neural network to compute the approximated posterior distribution $P(\theta|x_0)$ at the observed data $x_0$.
5. Compute the proposal prior:
    - If it is **truncated**, restrict the prior distribution to the support of the approximated posterior computed in step 4.
    - Otherwise, use this posterior distribution directly as the new proposal prior.
6. Update the prior distribution with the new proposal prior and return to step 1.

In the case of single-round inference, only steps 1 to 4 are performed.


!!! example

    An example of the following training and inference scripts for a single-round SBI is presented in
    `tutorials/tutorial_notebooks/08_learning_sbi_tutorial.ipynb`.

To perform our SBI using a dataset composed of heatmaps or 2D arrays of our synthetic pulsar populations (see 
[Generating density maps](generator_tutorial.md) for details) we use the script `pypopsyn/learning/sbi_train.py` as 
follows:

```commandline
python pypopsyn/learning/sbi_train.py --configuration tutorials/tutorial_notebooks/config_sbi.json
```

Here, the `config_sbi.json` file contains all the information required to optimize the neural network.

#### Folder Structure
First, we need to create the required folder structure for storing training and testing datasets 
for each round. Although this is not strictly necessary for single-round inference, we recommend doing it for consistency. 
You can do this by running the following command:

```commandline
python pypopsyn/learning/utils/data_folder_struct_sbi.py --base_path exp_folder_path
```


This will generate the following structure under `exp_folder_path`:

```commandline
exp_folder_path/
└── data/
    ├── test_dataset/
    │   ├── generated_dataset/
    │   │   └── round_0/
    │   └── simulations/
    └── training_dataset/
        ├── generated_dataset/
        │   └── round_0/
        └── simulations/
```

We recommend placing all files under the same base path. Specifically:

* Store the training statistics file in the `data/` directory.
* Save the training and testing datasets for the first round in: `data/training_dataset/generated_dataset/round_0/` and 
`data/test_dataset/generated_dataset/round_0/` respectively.


## General configuration options

We now discuss the various options in the `config_sbi.json` training configuration file, which include the type of SBI 
method, the type of compression if needed, the type of density estimator, the input shape of the dataset, and 
other relevant training hyperparameters.
  

#### General info

We first specify general settings for the experiment such as the experiment's name, the number of GPUs used, 
a manual seed for the initialization of the network weights (assuming that `set_manual_seed` is set to true) 
and some additional profiling options. The latter specify the names of the files containing run time information
for the code and whether this timing information is displayed in the terminal or not.

```json
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

#### Initialization

We also specify a scheme to initialize the weights and biases of the network. 
Here, we show an example using the Kaiming initializer denoted by `InitializerKaiming`.
```json
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

#### Density estimator

Next, we decide on the type of density estimator used to approximate the posterior distribution. The 
preconfigured options in the sbi library include so-called masked autoregressive flows `maf` or Gaussian mixture 
density networks `mdn`. For more details on these methods and relevant hyperparameters as well as custom density 
estimators see [here](https://sbi-dev.github.io/sbi/latest/tutorials/03_density_estimators/).

In the following example, we are setting a mixture density network with `10` Gaussian components, and the number of 
neurons in the hidden layers is set to `16`.

```json
{
    "density_estimator": {
        "type": "mdn",
        "args": {
            "num_components": 10,
            "hidden_features":16
        }
    },
}
```
#### MCMC sampler
For cases where Neural Ratio Estimation (NRE) or Neural Likelihood Estimation (NLE) is used, an extra step is required 
to sample from the posterior. In this case, we use the default MCMC-based sampling provided by the `sbi` package. 
For details on the different samplers in `sbi`, see [the sbi documentation](https://sbi-dev.github.io/sbi/latest/tutorials/09_sampler_interface/).

To perform MCMC sampling, you need to specify: the number of parallel chains, the thinning factor, 
and the MCMC sampler type.

`sbi` supports the following MCMC samplers: `nuts`, `slice`, `hmc`, and `slice_np_vectorized`.

```json
{ "mcmc_sampler": {
    "type": "slice_np_vectorized",
    "num_chains": 20,
    "thin": 5
  },
}
```
#### Model architecture

Next, we specify the architecture for the so-called embedding neural network. This embedding net is used to extract
features from the input data and compress the input into a latent vector that is then passed to the density estimator.
!!! note 
    This is only supported for SNPE, not for SNRE or SNLE.

In the following example, we will be using 2D maps as input and, hence, opt for a convolutional neural network (CNN) 
as the embedding net. This CNN is designed to adapt to any input size specified by the `input_shape` parameter and produce an output with a length specified by `len_output_layer`.
In this specific example the CNN receives an array of shape $32 \times 32$ with `3` different input channels 
(three of our density maps with a 32 resolution) as input and outputs a latent vector of size 32, which contains 
a compressed representation of the input feature maps.

The configuration file then looks as follows:
```json
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
The models that have been predefined are the following ones:

* `ModelConv` based on a CNN.
* `ModelLinear` based on a multi-layer perceptron (MPL) architecture.

If you would like to design your own network architecture, you need to implement a new model class in `pypopsyn/learning/models` and import this model in the file `models.py`.


#### Compression of input data
For cases where Neural Ratio Estimation (NRE) or Neural Likelihood Estimation (NLE) is used, an extra step is required 
to preprocess the data before passing it to the neural network. Unlike in Neural Posterior Estimation (NPE), you cannot train an embedding network 
simultaneously with the density estimator. To address this, we provide two options for data compression: a Convolutional
Neural Network (CNN) or Principal Component Analysis (PCA). The CNN is assumed to have been trained as part of a 
previous NPE experiment. The PCA model can be trained separately using the
`tutorials/analysis_notebooks/PCA_image_compressor.ipynb` notebook. 


```json
"compression_input": {
    "use_compression": true,
    "compression_type": "cnn",
    "cnn_model_path": "exp/models/SBI_ConvolutionMDNshallow/20250127_144110/round_0/trained_model.pickle",
    "pca_model_path":"/PCAs/1_pca_model_95_variance.pkl"
  },
```
Note that when `use_compression` is set to False, it can only be used with NPE. In this case, an embedding network will 
be trained simultaneously with the density estimator, and this embedding network will be responsible for compressing 
the input data.

#### Prior distribution
We need to specify the labels and prior ranges for plotting purposes. Note that the order of the list must match
the order of parameters in `dataset_full.csv`.
```json
"prior_ranges": {
          "labels":["B_initial_log10_mean", "B_initial_log10_sigma","P_initial_log10_mean", "P_initial_log10_sigma", "a_late", "L_radio_log10_mean", "epsilon_L"],
          "low": [12,0.1,-1.5,0.1,-3,24.6,0.1],
          "high":[14,1,0.5,1,0,28.6,1] 
          },
```
#### Training data loader

We also require a training data loader, which is responsible for loading the dataset in a representation readable 
by the network. The path to the directory containing the training dataset for the first round is specified in the `dataset_path_first_round`
field (specifically, a file named `dataset_full.csv`) and the JSON file characterizing the statistics of the training
dataset, and the input channels used in building our (multichannel) input. 
Additionally, we set the ground truth labels that we want to predict. For multi-round inference, the training datasets generated in each round will be saved in the directory specified by `dataset_path`.

Therefore, for the example above and following the recommended folder structure, the `training_data_loader` will look like this:

```json
 "training_data_loader": {
    "dataset_path": "exp_folder_path/data/training_dataset",
    "statistic_path": "exp_folder_path/data/statistics_train.json",
    "dataset_path_first_round":"exp_folder_path/data/training_dataset/generated_dataset/round_0",
    "filter_inputs": [9, 10, 11, 12, 13, 14],
    "filter_labels": [15, 16, 17, 18, 19, 20, 21],
    "normalize": false,
    "standardize": true,
    "num_sim":1000
  }
```

You must also specify, for the multi-round case, how many simulations to run in each round using the `num_sim` field for training.
In the example above, 1000 simulations are used for training.

Note that the datasets for the first round are expected to exist already for both multi-round and single-round inference. This is because the prior distribution in the 
first round is fixed, and these datasets can be reused across multiple experiments to save computational resources.


!!! note 

     When loading the data for SBI, we do not need to specify the batch size and the shuffle parameter.
     This is because the sbi library deals with shuffling the input dataset internally, while the batch size 
     is specified during the training procedure (see below). If we were to use the data loader as in the 
     [CNN learning tutorial](learning_tutorial_nn.md), it would load the dataset in a format that is not 
     compatible with sbi.

The available input channels and labels are specified in the `dataset_full.csv` file, where they are identified with 
an index starting from 0. Let us assume that our `dataset_full.csv` file looks as follows:
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

#### Testing data loader

Inference on a test dataset can be performed either separately using the `sbi_infer.py` script after training has been
completed, or simultaneously during training by setting `testing = true`. In the latter case, the testing dataset will
be generated on the fly, producing as many simulations as specified in `num_sim`.
As with training, testing requires specifying the path to the test dataset in the `dataset_path_first_round` field. 
This directory must contain a `dataset_full.csv` file. In the case of multi-round inference, the testing dataset will be 
saved at the path specified in `dataset_path`. Therefore, for the example above and following the recommended folder 
structure, the `test_data_loader` configurations will look like this:


```json
"test_data_loader": {
    "testing":false,
    "dataset_path": "exp_folder_path/data/test_dataset",
    "dataset_path_first_round":"exp_folder_path/data/test_dataset/generated_dataset/round_0",
    "num_sim":300},
```

#### Observed sample
The training script performs the inference for the observed sample specified by the `observed_sample` field. 
As before, you must provide the `filter_inputs` and `filter_labels`, which must match those used during training. 

```json
 "observed_sample":{
      "dataset_path": "data/example_generator_observed",
      "filter_inputs": [9, 10, 11, 12, 13, 14],
      "filter_labels": [15, 16, 17, 18, 19, 20, 21]
  },
```
#### Training parameters

Finally, we specify some general options for our machine learning experiment. First, we choose the SBI method to use, 
such as `snpe`, `snle`, or `snre`. For single-round inference, set `num_rounds = 1`. Additionally, you can configure 
the fraction of the training dataset reserved for validation, the training batch size, and the initial learning rate 
for the `Adam` optimizer (the default optimizer in the `sbi` library). 
You must also specify the directory path where the trained model will be saved. There is an option to train multiple
networks per round and create an ensemble of posteriors by setting `ensemble = true` and specifying the ensemble size 
with `size_ensemble`. Note that the difference between the networks in the ensemble comes only from the randomized
initialization of their weights.

```json
{
    "trainer": {
        "type": "snle",
        "validation_fraction": 0.1,
        "batch_size": 8,
        "lr": 5e-4,
        "ensemble":true,
        "size_ensemble":5, 
        "num_rounds":1,
        "save_dir": "data/example_learning_sbi"
    },
}
```

### Training output

The results of the training experiment are saved in the directory specified by the `save_dir` option. Specifically, 
training will create two folders in this directory, namely a `logs` folder and a `models` folder. Both contain 
subfolders for each specific training experiment of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes 
the date and time when the experiment was launched. Within that folder there will be one folder per each round, in the 
case of single round there will be just one folder called `round_0` 

The `logs` directory contains the files `profile.json` and `profile.log`, which provide timing and profiling information for the inference script executed in each round.
Additionally, within each `round_{i}` subfolder inside the logs directory, the following files are included:

* `training_statistics_{j}.json` with the training and validation loss evolution.
* `training_stats_{j}.pdf` with a plot showing the training and validation loss evolution.

In this example, `j` indicates which neural network in the ensemble the file refers to.

Finally, each subfolder `round_{i}` in the `models` directory contains:

* `trained_model.pickle`: Contains the trained model. This object is needed to sample from the approximated posterior 
   distribution after the neural network has been trained.
* `inference.pickle`: Contains the trained inference object, which stores the weights of the trained neural network. 
* `samples_posterior.pt`: A tensor containing samples from the posterior distribution conditioned on the observed data.
* `corner_plot_observed_sample.pdf`: A corner plot visualizing the approximated posterior distribution conditioned on 
   the observed data.
* `posterior_samples_test_data.npz`: A tensor containing the true values and the corresponding posterior samples for 
   each sample in the test dataset.

If `ensemble` is enabled, there will be as many `inference.pickle` and `trained_model.pickle` files as there are neural 
networks in the ensemble.

## Multi-round specific options
This section describes configuration options specific to multi-round inference.

### Resume mode

Resume mode allows training to continue from a previous round if it was interrupted before completion. To use resume 
mode:

* Set `config["resume_training"]["resume"] = true`.
* Specify the last completed round to resume from using `config["resume_training"]["last_round"] = 2`.
* Provide the paths to the previously saved model and logs using: `config["resume_training"]["save_dir"]` and 
  `config["resume_training"]["log_dir"]` respectively.

This setup ensures continuity in output files and prevents the creation of a new directory for resumed runs.  
We recommend making a copy of the learning directory to avoid overwriting data from previous rounds and to verify 
the consistency of the resume mode.
When resuming, in the first iteration, you need to load the trained model and the training dataset from the last 
completed round. This is necessary to:

* Compute the proposal prior distribution for the next round.
* Load the training dataset, since simulations from previous rounds are reused in subsequent rounds 
  when `append_simulations` is enable.

An example of the configuration file:

```json
 "resume_training": {
    "resume": false,
    "last_round": 3,
    "save_dir": "exp/learning/models/SBI_ConvolutionMDN/20240705_123828",
    "log_dir": "exp/learning/models/SBI_ConvolutionMDN/20240705_123828"
  },
```

In this example, we will load the `inference.pickle` and `trained_model.pickle` from round 3 and compute the
approximated posterior distribution at the observed sample, which will serve as the proposal prior for the next round.
From round 4 onward, the computation will proceed as usual in a multi-round inference approach.


### Extra parameter for training
Several additional parameters control how training behaves across rounds:

* `append_simulations`: If set to `true`, simulations from previous rounds are included in the current round.

* `truncated_prior`: If set to `true`, the proposal prior is obtained by truncating the initial prior with the posterior
  from the previous round (evaluated at the observed data). Otherwise, the proposal prior is simply the 
 approximated posterior distribution from the previous round.

* `sir`: If set to `true`, Sampling Importance Resampling (SIR) is used for truncated prior sampling. Otherwise, 
 rejection sampling is used. For an explanation of these two methods, we refer the user to [Liu, J. S. (2001), Monte Carlo Strategies in Scientific Computing.](https://github.com/szcf-weiya/MonteCarlo/blob/master/References/Monte-Carlo-Strategies-in-Scientific-Computing.pdf)

* `retrain_from_scratch`: If set to `true`, the model is retrained from scratch in each round, i.e., the model weights are re-initialized in each round. 
  Otherwise, training continues updating the weights trained in the previous rounds.

* `plot_proposal`: If set to `true`, a corner plot of the proposal prior will be saved.

Note: For SNPE, you cannot enable both `truncated_prior = false` and `append_simulations = true`. 

Example trainer block:

```json
"trainer": {
    "type": "snle",
    "append_simulations":true,
    "truncated_prior": false,
    "retrain_from_scratch": true,
    "ensemble":true,
    "size_ensemble":5, 
    "validation_fraction": 0.1,
    "batch_size": 16,
    "lr": 1e-4,
    "num_rounds":10,
    "sir":true,
    "plot_proposal": false,
    "save_dir": "/data/magnesia/common/paper_pardo_araujo_etal_2025/exp_constant_mag_cnn_embedding/learning"
  },
```



### Running simulation in parallel for each round

There are two methods to parallelize the simulation process: using either the [`dask`](https://www.dask.org/) or the
[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) packages.

- To use Dask, set `enable_dask` to `true` in the configuration file and specify the number of workers with `n_workers`
  (this corresponds to the number of HTCondor jobs on the PIC server).
- If Dask is not enabled, the `multiprocessing` package will be used instead. You can specify the number of parallel
  processes via: `"n_processes": <num_processes>`

### Notes on Running with Dask on the PIC Server

There are some important considerations for MAGNESIA users when running simulations with Dask on the PIC server:

#### Folder Handling
- The script copies the `MAGNESIA_population_synthesis` folder to each node to avoid redundant reads and reduce server load.
- You must update the following in `pypopsyn/simulator/config_simulator.py`:
    ```python
    cfg["server_run"] = False
    path_to_software = "MAGNESIA_population_synthesis"
    ```
  Each node will have its own copy of this folder and will access files locally.

#### Dask Usage on the PIC Server
To reduce file system load, the simulation output folder is created and saved locally on each node.  After the 
simulation completes, the output folder is copied back to the original dataset path, specified in
`config["training_data_loader"]["dataset_path"]`.  This logic is implemented in the function `run_simulation_dask` 
located in `utilities/simulation_helper/run_simulation_set.py`.

#### Running the Main Job on HTCondor
 To run the main job on HTCondor, add the following lines to your submit file:

    ```bash
    RUN_FOLDER = <experiment_folder>
    LOG_FOLDER = $(RUN_FOLDER)/logs
    initialdir = $(RUN_FOLDER)
    remote_initialdir = $(RUN_FOLDER)
    
    universe        = vanilla 
    executable      = <experiment_folder>/htcondor_submit/wrapper.sh
    log             = <experiment_folder>/htcondor_output/$(ProcId)-log.txt
    output          = <experiment_folder>/htcondor_output/$(ProcId)-out.txt 
    error           = <experiment_folder>/htcondor_output/$(ProcId)-error.txt 
    
    request_gpus=1
    
    Queue
    ```

!!!Note 
    In the example above, we assume you have created `htcondor_submit` and `htcondor_output` folders. The first is used to store 
    the `job.submit` and  `wrapper.sh` files, and the second stores the `stdout` and `stderr` from the main job.

We recommend saving the `htcondor_submit` and `htcondor_output` folders within the same `exp_folder_path` directory where the
`data` folder is located (see section [Folder Structure](#folder-structure)).

To read more general documentation about HTCondor, refer to the [HTCondor documentation](../basics/HTCondor.md).
#### Monitoring Dask Workers (HTCondor + GPU)

If this is your first time using Dask on the PIC server, you must manually launch an empty Dask cluster via the Jupyter
dashboard (located in the left sidebar on the PIC Jupyter interface) to ensure correct functionality. If you're running the main job on HTCondor with GPU, you can monitor the 
Dask workers through Jupyter following these steps:

   1. Open the Jupyter interface on a GPU node.
   2. Access the stdout log of the main job:
      - Run `condor_q` to find the `job_id`. This job should **not** have `HTCondorCluster` as the batch name.
      - Use `condor_ssh_to_job <job_id>` to SSH into the job.
      - Navigate to the home directory (`cd`) and run `cat _condor_stdout` to view the stdout log.
   3. Look for a line similar to:
     ```
     INFO:distributed.scheduler: dashboard at: http://192.168.100.78:8787/status
     ```
4. Extract the port number (e.g., `8787`) and open the Dask dashboard via the Jupyter interface (left sidebar on the PIC 
   Jupyter interface) by pasting the following URL into your browser (replace `<your_username>` and `<port>` 
   accordingly):

      ```
      https://jupyter.pic.es/user/<your_username>/proxy/<port>
      ```
   


## Inferring on a dataset

Once our SBI pipeline has been trained, it can be used to infer on an unseen dataset of generated maps and extract 
posterior distributions of the corresponding pulsar
population parameters. The script `pypopsyn/learning/sbi_infer.py` allows us to take an experiment configuration file, 
a pre-trained model, and a dataset, and run the inference.

### Configuration options

In addition to the testing data loaders specified above, we also need to define the directories for saving
and loading inference results using the `save_dir` and `load_dir` parameters under the `infer` field in the configuration file. 
The `load_dir` should point to the location where `inference.pickle` and `trained_model.pickle` are stored, as these 
files are required to perform inference.
You must also specify whether to compute the coverage probability using the `compute_coverage` flag.
If `compute_coverage` is not enabled, the output will include a corner plot and samples from the posterior distribution 
conditioned on the observed data. If it is enabled, additional outputs will be saved:

* `posterior_samples_test_data.npz`: containing the true values and posterior samples for each sample of the test dataset.
* `coverage_probability.npy`: a tensor of coverage probabilities.
* `coverage_plot.pdf`: a plot visualizing the coverage probabilities.

If `sim_dataset` is set to `True`, the test dataset will be generated in each round. Otherwise, it is assumed that the 
directory specified in the `dataset_path` field under `test_dataset_loader` contains a pre-generated test dataset for each 
round.

```json
{
    "infer": {
    "sim_dataset":false ,
    "compute_coverage":true,
    "load_dir": "exp_1/learning/models/SBI_ConvolutionMDN/20250403_192124",
    "save_dir": "exp_1/inference"
  }
}
```
!!! note
    The `sbi_infer.py` script requires the folder structure explained above to function correctly and to properly load the test dataset.

Once the inference configuration is set up, we run the inference script by providing the configuration file 
(`--configuration`) as follow:
```commandline
python pypopsyn/learning/sbi_infer.py --configuration tutorials/tutorial_notebooks/config_sbi.json 
```
### Inference output

If the inference is successful, the output of `pypopsyn/learning/sbi_infer.py` will be saved in the directory 
specified in the `save_dir` option. Specifically, inference will create a `logs` folder in this directory containing
subfolders of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes the date and time when the inference
was launched.

The `logs` directory contains the files `profile.json` and `profile.log`, which provide timing and profiling information 
for the inference script executed in each round. Additionally, within each `round_{i}` subfolder inside the logs 
directory, the following files are included:

* `samples_posterior.pt`: A tensor containing samples from the posterior distribution conditioned on the observed data.
* `corner_plot_observed_sample.pdf`: A corner plot visualizing the approximated posterior distribution conditioned on 
   the observed data.
* `posterior_samples_test_data.npz`: A tensor containing the true values and the corresponding posterior samples for 
   each sample in the test dataset.
* `coverage_plot.pdf` and `coverage_probability.npy` files with the results of the coverage probability diagnostic test.

