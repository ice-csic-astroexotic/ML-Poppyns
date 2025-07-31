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

Using the `pypopsyn/learning/sbi_train.py` script, we support the following approaches for SBI:

 1. Neural Posterior Estimation ([NPE](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)): A neural network is trained to approximate the posterior distribution 
    directly, learning a mapping from model parameters $\theta$ to $P(\theta | x)$.

 2. Neural Likelihood Estimation ([NLE](https://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)): A neural network emulates the simulator by approximating the likelihood 
    $P(x | \theta)$. Once trained, this model can be used with standard sampling algorithms (e.g., MCMC) to compute or 
    sample from the posterior.

 3. Neural Ratio Estimation ([NRE](https://proceedings.mlr.press/v119/hermans20a/hermans20a.pdf)): A classifier is trained to approximate the likelihood-to-evidence ratio
    $r(\theta, x) = P(x | \theta) / P(x)$, which can be used to compute or sample from the posterior using methods 
    like MCMC.

## Amortized vs. sequential inference

SBI can be performed using either amortized or sequential strategies. While amortized inference enables fast posterior 
estimates for any input after a single large training phase (because inference simply corresponds to a single forward
pass through the network), sequential methods focus the simulation budget on regions most relevant to explaining a 
specific observation. This advantage makes sequential methods more efficient for application cases where only a single 
(or small) observed sample is present and the simulator is expensive, as is the case for our application. 

In the following, we will refer to amortized as **single-round** and sequential as **multi-round** inference, with the 
former being a limiting case of the multi-round approach. The three methods summarised above have their own sequential 
variants, which are named by adding an 'S' at the beginning. For example, SNPE stands for Sequential Neural Posterior 
Estimation. 

For multi-round inference, the workflow is as follows:

1. Sample the proposal prior distribution to obtain $\theta_i \sim P(\theta)$.
2. Given the parameter samples from step 1, generate synthetic data $x_i \sim P(x|\theta_i)$ using the simulator.
3. Train the neural network on the training dataset consisting of pairs $(\theta_i, x_i)$ obtained in the previous 
   steps.
4. Use the trained neural network to compute the approximated posterior distribution $P(\theta|x_0)$ at the observed 
   data $x_0$.
5. Compute a new proposal prior using the following options:
    - If we opt for a **truncated approach**, restrict the prior distribution to the support of the approximated 
      posterior computed in step 4. For more information about truncated proposal priors, see 
      [Deistler et al. (2022)](https://arxiv.org/abs/2210.04815) and 
      [Pardo-Araujo et al. (2025)](https://www.aanda.org/articles/aa/pdf/2025/04/aa53314-24.pdf).
    - Otherwise, use the posterior distribution from step 4 directly as the new proposal prior.
6. Update the prior distribution with the new proposal prior and return to step 1.

In the case of single-round inference, only steps 1 to 4 are performed.

!!! example

    An example of NPE with single-round SBI using the training and inference scripts discussed in detail below
    is presented in `tutorials/tutorial_notebooks/08_learning_sbi_tutorial.ipynb`.

## Training a SBI pipeline

### Folder structure

Before executing the SBI training, we need to create the required folder structure for storing training and testing 
datasets for each round. Although this is not strictly necessary for single-round inference, we recommend doing it 
for consistency. We obtain the relevant folder structure by running the following command:

```commandline
python pypopsyn/learning/utils/data_folder_struct_sbi.py --base_path output/
```

This will generate the following structure under `output`:

```commandline
output/
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

We recommend placing all files relevant for training in the same base path. Specifically,

* Store the training statistics file in the `data/` directory.
  * Save the training and testing datasets for round 0 in `data/training_dataset/generated_dataset/round_0/` and 
  `data/test_dataset/generated_dataset/round_0/`, respectively.

!!! note
    When running training or inference in the single-round mode, the output is still saved using the multi-round
    structure. That is, results will be stored in a folder named `round_0`.

### Main training script

To perform our SBI using a dataset composed of heatmaps or 2D arrays of our synthetic pulsar populations (see 
the tutorial [Generating density maps](generator_tutorial.md) for details), we execute the script `pypopsyn/learning/sbi_train.py` as 
follows:

```commandline
python pypopsyn/learning/sbi_train.py --configuration tutorials/tutorial_notebooks/config_train_sbi.json
```

Here, the `config_train_sbi.json` file contains all the information required to optimize the neural network.

### Configuration options

We now discuss the various options in the `config_train_sbi.json` SBI configuration file, which include the type 
of SBI method, the type of compression if needed, the type of density estimator, the input shape of the dataset, 
and other relevant training hyperparameters used. Note that several of these are equivalent to those outlined in 
the tutorial [Learning pulsar parameters with NNs](learning_tutorial_nn.md), where we discuss point estimation. In the following, we
primarily focus on single-round inference as outlined in the tutorial `08_learning_sbi_tutorial.ipynb`. We explain 
additional parameters specific to multi-round inference further below in [Multi-round specific options](#multi-round-specific-options).

#### General info

We first specify general settings for the experiment such as the experiment's name, the number of GPUs used, whether 
to fix the random seed (and its value) and some additional profiling options. The latter specify the names of the files
containing run time information for the code and whether this timing information is displayed in the terminal or not. 
We discuss the multi-processing options in detail in [Running simulation in parallel for each round](#running-simulation-in-parallel-for-each-round).

```json
{
"name": "SBI_ConvolutionMDN",
    "n_gpu": 1,
    "enable_dask": false,
    "n_processes": 4,
    "workers_dask": 4,
    "set_manual_seed": false,
    "manual_seed": 42,
    "profile_log": "profile.log",
    "profile_json": "profile.json",
    "show_profiling": true
}
```

#### Training parameters

General options for the machine learning experiment are specified in the trainer section. First, we select the SBI 
method to use by choosing from `snpe`, `snle`, or `snre` (including the s for the sequential approach, even in the case 
of single-round inference). For single-round inference, we set `num_rounds = 1`; otherwise, this parameter defines the 
number of rounds to perform.
We must also specify the directory where the trained model will be saved using the `save_dir` parameter. Additional 
configuration options include the fraction of the training dataset reserved for validation, the training batch size, 
and the initial learning rate for the Adam optimizer, which is the default in the `sbi` library.

To create an ensemble of posteriors, we can train multiple networks per round by setting `ensemble = true` and specifying
the ensemble size using `size_ensemble`. Note that the networks in the ensemble share the same architecture and differ 
only due to the random initialization of their weights. See [Multi-round specific options](#multi-round-specific-options)
section below for more information on the last parameters, which are specific to multi-round inference.

```json
{
    "trainer": {
        "type": "snpe",
        "num_rounds": 1,
        "save_dir": "output/learning_sbi",
        "validation_fraction": 0.1,
        "batch_size": 8,
        "lr": 0.0005,
        "ensemble": false,
        "size_ensemble": 5,
        "truncated_prior": false,
        "retrain_from_scratch": false,
        "sir": true,
        "append_simulations": true,
        "plot_proposal": false
      
    }
}
```
#### Density estimator

Next, we decide on the type of density estimator used to approximate the posterior distribution. The 
preconfigured options in the `sbi` library include so-called masked autoregressive flows `maf` or Gaussian mixture 
density networks `mdn`. We can set the number of hidden features for the density estimator.
When using `mdn` or `maf`, we can also set the number of components in the mixture or the number of transformations in the 
flow, respectively, using the `num_components` and `num_transforms` parameters. On the other hand, for SNRE, we can 
specify the type of classifier to use, such as: `linear`, `mlp`, or `resnet`. For more details on these methods and 
relevant hyperparameters as well as custom density estimators see [here](https://sbi-dev.github.io/sbi/latest/tutorials/03_density_estimators/).

In the following example, we are setting a mixture density network with `10` Gaussian components, and the number of 
neurons in the hidden layers is set to `32`.

```json
{
  "density_estimator": {
    "type": "mdn",
    "classifier_nre": "resnet",
    "args": {
      "hidden_features": 32,
      "num_components": 10, 
      "num_transforms": 5
    }
  }
}
```
If the model type selected in the training options is not `snre`, and the density estimator is not `maf`, then the classifier and the num_components specified here will be ignored.


#### MCMC sampler

For cases where NRE or NLE is used, an extra step is required to obtain or sample from the posterior. In the following 
case, we use the default MCMC-based sampling algorithms provided by the `sbi` package. For details on the different 
samplers in `sbi`, see [the official documentation](https://sbi-dev.github.io/sbi/latest/tutorials/09_sampler_interface/). In the case of NPE, the following options are simply ignored.

When performing MCMC sampling, we need to specify the number of parallel chains, the thinning factor, and the MCMC 
sampler type. `sbi` supports the following MCMC samplers: `nuts`, `slice`, `hmc`, and `slice_np_vectorized`.

```json
{ 
  "mcmc_sampler": {
    "type": "slice_np_vectorized",
    "num_chains": 20,
    "thin": 5
  }
}
```

#### Embedding network

Next, we specify the architecture for the so-called embedding neural network. This embedding net is used to extract
features from the input data and compress the input into a latent vector that is then passed to the density estimator.
This neural network is optimised at the same time as the parameters of the neural density estimator.

!!! note 
    This functionality is only supported in SNPE, but not available for SNRE or SNLE. This is because, unlike in NPE, 
    the neural density estimators in NRE and NLE approximates the likelihood or the likelihood ratio
    directly. As a result, the output of the neural network must match those of the simulator, meaning the 
    compression step would have to occur after the density estimator, which is not possible.

In the following example, we will be using 2D maps as input and, hence, opt for a convolutional neural network (CNN) 
as the embedding net. This CNN is designed to adapt to any input size specified by the `input_shape` parameter and 
produce an output with a length specified by `len_output_layer`. 

There are currently three predefined models available for the embedding network:

1. `ModelConvSBI`: a CNN with three convolutional layers.

2. `ModelConvSBIdeep`: a deeper CNN with four convolutional layers.

3. `ModelConvSBIshallow`: a shallower CNN with two convolutional layers.

Therefore, when using ModelConvSBI, the configuration file looks as follows:

```json
{
    "arch": {
        "type": "ModelConvSBI",
        "args": {
            "input_shape": [3, 32, 32],
            "len_output_layer": 32
        }
    }
}
```
If you would like to design your own network architecture, you need to implement a new model class in 
`pypopsyn/learning/models` and import this model in the file `models.py`.

#### Initialization

We also specify a scheme to initialize the weights and biases of the convolutional neural network. 
Here, we show an example using the Kaiming initializer denoted by `InitializerKaiming`.

```json
{
    "weights_initializer": {
        "type": "InitializerKaiming",
        "args": {}
    }
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

#### Alternative input compression options

For learning approaches where NRE or NLE are applied, as outlined above we cannot train an embedding network 
simultaneously with the density estimator but instead apply another step to preprocess the data before passing it 
to the neural network. For this purpose, we provide two options for separate data compression: a Convolutional Neural 
Network (CNN) or Principal Component Analysis (PCA). Note that our implementation assumes that the CNN corresponds to 
a fully optimised (i.e., fixed) embedding network that has been trained as part of a previous NPE experiment. For 
both PCA and CNN, the code expects a pickle file containing the pretrained model. The PCA model can be trained 
separately using the `tutorials/analysis_notebooks/PCA_image_compressor.ipynb` notebook. 

```json
{
  "compression_input": {
    "use_compression": true,
    "compression_type": "cnn",
    "cnn_model_path": "exp_4/round_9/trained_model_ensemble_0.pickle",
    "pca_model_path": "PCAs/1_PCA_95_new/pca_model_combined.pkl"
  }
}
```
Note that when `use_compression` is set to False, the training configuration can only be used with NPE. In this case, 
an embedding network will be trained simultaneously with the density estimator as outlined in the previous section, and 
this embedding network is responsible for compressing the input data.

#### Prior distribution for training data

In order to perform multi-round inference with a truncated approach or visualise cornerplots for our inference results
across all SBI methods, we need to specify the labels and prior ranges (corresponding to the initial prior ranges which
are subsequently restricted in the TSNPE algorithm). Note that the order of the list must match the order of parameters 
in the `dataset_full.csv` file that contains information on the training data.

```json
{
  "prior_ranges": {
    "labels": [
      "B_initial_log10_mean",
      "P_initial_log10_mean"
    ],
    "low": [12, -1.5],
    "high": [14, 0.3]
  }
}
```

#### Training data loader

We also require a training data loader, which is responsible for loading the dataset in a representation readable 
by the network. The path to the directory containing the training dataset (in particular the file `dataset_full.csv`) 
for round 0 is specified in the `dataset_path_round_0` field. For multi-round inference, the training
datasets generated in each of the following rounds will then be saved in the directory specified by `dataset_path`.
We also specify the location for the JSON file characterizing the statistics of the training dataset in `statistic_path`
and the input channels `filter_inputs` used in building our (multichannel) input. Additionally, we set the ground truth
labels `filter_labels` that we want to predict and provide information on whether our data and labels are normalized or
standardized. The `filter_labels` and `filter_inputs` arguments are explained in more detail below, along with an 
example.

For the multi-round case, we must also specify, how many simulations we want to run in each round using the `num_sim` 
field for training. In the example below, 1000 simulations are generated per round and then used for training.

We also note that our workflow for both multi-round and single-round inference assumes that the dataset for the first 
round already exist. This is possible because the prior distribution in round 0 is fixed, and these datasets
can be reused across multiple experiments to save computational resources.

For the example above and following the recommended folder structure, the `training_data_loader` will look like this:

```json
{
 "training_data_loader": {
    "dataset_path_round_0": "output/data/training_dataset/generated_dataset/round_0",
    "dataset_path": "output/data/training_dataset",
    "statistic_path": "output/data/statistics_train.json",
    "filter_inputs": [9, 10, 11],
    "filter_labels": [15, 17],
    "normalize": false,
    "standardize": true,
    "num_sim": 1000
  }
}
```

!!! note 

     When loading the data for SBI, we do not need to specify the batch size and the shuffle parameter.
     This is because the `sbi` library deals with shuffling the input dataset internally, while the batch size 
     is specified during the training procedure (see above). If we were to use the data loader as in the 
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
`survey_PMPS_ppdot_map`, `survey_SMPS_ppdot_map`, `survey_HTRU_ppdot_map` (indices 9, 10, 11) and want to predict
the labels `B_initial_log10_mean` and `P_initial_log10_mean` (indices 15, 16).

!!! warning
    
    If an embedding network is used to compress the 2D maps, the length of `filter_inputs` in the dataset loader 
    configuration must match the network’s expected input channel dimension specified in `input_shape` in the `arch` 
    section.

Finally, our training data loader enables us to activate on-the-fly normalization or standardization (both are 
mutually exclusive) for the input maps and ground truths (labels). Both take advantage of the statistical information 
contained in the `statistics_train.json` file. For normalization, the input channels and labels will have values in 
the range between 0 and 1. If standardized, the input channels and labels have values centred around 0 and range 
approximately between -1 and 1.

!!! note

    The training dataset is also used for validation. This is automatically handled through sbi. We therefore do
    not require a separate validation data loader. We can specify the corresponding validation fraction as outlined 
    below.

#### Testing data loader

Inference on a test dataset can be performed either separately using the `sbi_infer.py` script after training, or 
simultaneously during training by setting the `testing` field to true. As with training, testing requires specifying 
the path to the test dataset for round 0 in the `dataset_path_round_0` field, which must contain a 
`dataset_full.csv` file. In multi-round inference, the test dataset for subsequent rounds is generated on the fly and 
saved to the path specified in `dataset_path`, with the number of simulations defined by `num_sim`.

Following the recommended folder structure, the `test_data_loader` configuration for the example above looks as follows
(although we have opted to not perform testing simultaneously here and set `testing` to false:

```json
{
  "test_data_loader": {
    "testing": false,
    "dataset_path_round_0": "output/data/test_dataset/generated_dataset/round_0",
    "dataset_path": "output/data/test_dataset/generated_dataset",
    "num_sim": 10
  }
}
```

#### Observed sample

The inference script performs inference on a sample specified in the `observed_sample` field, assuming that the folder 
contains a CSV file named `dataset_atnf.csv`. As before, you must provide `filter_inputs` and `filter_labels`, which
must match those used during training.

```json
{
  "observed_sample": {
      "dataset_path": "data/example_generator_observed",
      "filter_inputs": [9, 10, 11],
      "filter_labels": [15, 17]
  }
}
```
#### Dynamical database

To perform the magneto-rotational evolution, we first need to sample stars from a dynamical database and then carry out 
the magneto-rotational evolution. In the `dyn_data_loader`, we specify the path to the dynamical database, assuming that 
it contains a file named `final_pop_dyn.csv`. This file can be generated using the tutorial notebook located at 
`tutorials/tutorial_notebooks/02_simulator_dyn_tutorial.ipynb`.

```json
{
  "dyn_data_loader": {
      "dataset_path": "../../data/example_simulation_dyn"
    }
}
```

#### Training output

The results of the training experiment are saved in the directory specified by the `save_dir` option outlined under
[Training parameters](#training-parameters) above. Specifically, training will create two folders in this directory, 
namely a `logs` folder and a `models` folder. Both contain subfolders for each specific training experiment of the
form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes the date and time when the experiment was launched.
Within these folders, the training experiment will result in the creation of one subfolder for each round of training.
In the case of single-round inference, we would only obtain one folder named `round_0`. 

The `logs` subdirectories contain the files `profile.json` and `profile.log`, which provide timing and profiling 
information for the training script executed across all rounds. Additionally, each `round_{i}` subfolder inside the 
`logs` directory contains the following files:

* `training_statistics_{j}.json` with the training and validation loss evolution.
* `training_stats_{j}.pdf` with a plot showing the training and validation loss evolution.

The `j` indicates which neural network in the ensemble the file refers to. In the case of using just one neural network
no subscript is added to these two files. 

Finally, each subfolder `round_{i}` in the `models` directory contains:

* `corner_plot_observed_sample.pdf`: A corner plot visualizing the approximated posterior distribution conditioned on 
   the observed data.
* `coverage_plot.pdf` and `coverage_probability.npy` files with the results of the coverage probability diagnostic
   test if testing was enabled.
* `inference.pickle`: The trained inference object, which stores the weights of the trained neural network. Knowledge
   of these are required when resuming training in the multi-round case, where we do not want to start from scratch 
   but continue from the last trained network.
* `posterior_samples_test_data.npz`: A NumPy array containing the true values and the corresponding posterior samples
   for each sample in the test dataset if testing is enable .
* `samples_posterior.pt`: A tensor containing samples from the posterior distribution conditioned on the observed data.
* `trained_model.pickle`: The trained model. This object is needed to sample from the approximated posterior 
   distribution after the neural network has been trained.

If `ensemble` is enabled, there will be as many `inference.pickle` and `trained_model.pickle` files as there are 
neural networks in the ensemble.

### Multi-round specific options

This section describes configuration options specific to multi-round inference.

#### Training specific parameters for multi-round inference

Several additional parameters control how training behaves across different rounds:


* `truncated_prior`: If set to `true`, the proposal prior is obtained by truncating the initial prior with the
   posterior from the previous round (evaluated at the observed data, 
    [Deistler et al. 2022](https://arxiv.org/abs/2210.04815)). Otherwise, the proposal prior is simply the 
   approximated posterior distribution from the previous round.

* `retrain_from_scratch`: If set to `true`, the neural network is retrained from scratch in each round, i.e., the 
   model weights are re-initialized in each round. Otherwise, training continues updating the weights trained in the
   previous rounds.

* `sir`: If set to `true`, Sampling Importance Resampling (SIR) is used for truncated prior sampling. Otherwise, 
   rejection sampling is used. Note that rejection sampling can be significantly more computationally expensive if the 
   posterior distribution is very narrow and the rejection region is large. For an explanation of these two methods, 
   we refer the user to
   [Liu, J. S. (2001), Monte Carlo Strategies in Scientific Computing.](https://github.com/szcf-weiya/MonteCarlo/blob/master/References/Monte-Carlo-Strategies-in-Scientific-Computing.pdf)

* `append_simulations`: If set to `true`, simulations from previous rounds are included in the current round. 
   ([Deistler et al. 2022](https://arxiv.org/abs/2210.04815)).

* `plot_proposal`: If set to `true`, a corner plot of the proposal prior will be saved.

!!! warning
    For SNPE, we cannot enable both `truncated_prior = false` and `append_simulations = true` simultaneously because in 
    this scenario the approximate posterior must be corrected using the proposal prior from the corresponding round 
    (Appendix 6.2 of [Deistler et al. 2022](https://arxiv.org/abs/2210.04815)).

An example of the multi-round training information could look as follows:

```json
{
  "trainer": {
        "type": "snpe",
        "num_rounds": 2,
        "save_dir": "output/learning_sbi",
        "validation_fraction": 0.1,
        "batch_size": 8,
        "lr": 0.0005,
        "ensemble": false,
        "size_ensemble": 5,
        "truncated_prior": false,
        "retrain_from_scratch": false,
        "sir": true,
        "append_simulations": true,
        "plot_proposal": false
}
}
```
#### Resume mode

Resume mode allows us to continue the SBI training from a previous round, as might, e.g., be necessary if the training
was interrupted in an earlier run before it was completed. The following set-up will ensure continuity in the output 
files and prevents the creation of a new directory for resumed runs. However, we recommend making a copy of the 
existing `learning` directory to avoid overwriting data from previous rounds and to verify the consistency of 
resumed runs.

To use the resume mode we first need to:

* Set `config["resume_training"]["resume"] = true`.
* Specify the last completed round to resume from, e.g., using `config["resume_training"]["last_round"] = 7`.
* Provide the paths to the previously saved model and logs using `config["resume_training"]["save_dir"]` and 
  `config["resume_training"]["log_dir"]`, respectively.

When resuming, the first (new) iteration requires loading the trained model from the 
previously completed round of an earlier experiment. This is necessary to compute the proposal prior distribution 
for the next (first new) round. Note that if `append_simulations` is set to true, simulations from all previous rounds 
are reused at every round. Therefore, when resuming, it's essential to load the training datasets
from all completed rounds.

An example of the configuration file that enables resuming would look as follows:

```json
{
  "resume_training": {
  "resume": true,
  "last_round": 7,
  "save_dir": "exp/learning/models/SBI_ConvolutionMDN/20240705_123828",
  "log_dir": "exp/learning/models/SBI_ConvolutionMDN/20240705_123828"
  }
}
```

In this example, we will load the `inference.pickle` and `trained_model.pickle` from round 7 saved in the `save_dir`
folder and compute the approximated posterior distribution at an observed sample, which will then serve as the proposal 
prior for the next round. From round 8 onward, the computation will proceed as usual in our multi-round inference 
approach. The `inference.pickle` and the `trained_model.pickle` files for the `last_round` and onward will be saved in 
the same folder specify in `save_dir`. The logs and training statistics are saved in the `log_dir` folder.

#### Running simulation in parallel for each round

We have implemented two ways to parallelize the simulation process to optimise computation time: using either the 
[`Dask`](https://www.dask.org/) or the [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) packages.

The desired parallelization method is specified at the top of the `config_train_sbi.json` file in the 
[General info](#general_info) section. It, for example, looks as follows:

```json
{
  "name": "SBI_ConvolutionMDN",
  "n_gpu": 1,
  "enable_dask": false,
  "n_processes": 4,
  "workers_dask": 4
}
```

- To use Dask, we set `enable_dask` to true and specify the number of workers with `n_workers`, which corresponds to 
   the number of HTCondor jobs on the PIC server.
- If Dask is not enabled, the `multiprocessing` package will be automatically used instead. We can specify the 
   number of parallel processes via `n_processes`.

#### Notes on running Dask on the PIC

There are some important considerations for MAGNESIA users when running simulations with Dask on the PIC server:

##### Folder handling

- During training the `MAGNESIA_population_synthesis` folder is copied to each node to avoid redundant reads and reduce 
   server load.
- You must update the following in `pypopsyn/simulator/config_simulator.py` as each node will have its own copy of 
   the full code repository and will access files locally:
    ```python
        cfg["server_run"] = False
        path_to_software = "MAGNESIA_population_synthesis"
    ```

##### Dask usage on the PIC server

To reduce file system load, the simulation output folder is created and saved locally on each node.  After the 
simulation completes, the output folder is copied back to the original dataset path, specified in
`config["training_data_loader"]["dataset_path"]`.  This logic is implemented in the function `simulator_dask` 
located in `utilities/experiment_helpers/run_simulation_set_sbi.py`.

##### Running the main job using HTCondor

While we refer to the [HTCondor documentation](../basics/HTCondor.md) for general information about HTCondor, we 
highlight a special aspect of multi-round inference executed with HTCondor in combination with Dask here. To run the 
main job on HTCondor, the submit file has to have the following content:

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
    In the example above, we assume that we have created `htcondor_submit` and `htcondor_output` folders. 
    The first is used to store the `job.submit` and  `wrapper.sh` files, and the second stores the `stdout` 
    and `stderr` from the main job.

We recommend saving the `htcondor_submit` and `htcondor_output` folders within the same `output` directory where 
the `data` folder is located (see [Folder structure](#folder-structure) above for details).

##### Monitoring Dask workers (HTCondor + GPU)

If this is your first time using Dask on the PIC server, you must manually launch an empty Dask cluster via the Jupyter
dashboard (located in the left sidebar on the PIC Jupyter interface) to ensure correct functionality. If you are
running the main SBI training job on HTCondor with GPUs, you can monitor the Dask workers through Jupyter following
these steps:

   1. Open the Jupyter interface on a GPU node.
   2. Access the stdout log of the main job as follows:
      - Run `condor_q` to find the `job_id`. This job should **not** have `HTCondorCluster` as the batch name.
      - Use `condor_ssh_to_job <job_id>` to SSH into the job.
      - Navigate to the home directory (`cd`) and run `cat _condor_stdout` to view the stdout log.
   3. Look for a line similar to:
      ```
      INFO:distributed.scheduler: dashboard at: http://192.168.100.78:8787/status
      ```
   4. Extract the port number (e.g., `8787`) and open the Dask dashboard via the Jupyter interface (left sidebar on 
      the PIC Jupyter interface) by pasting the following URL into your browser (replace `<your_username>` and
      `<port>` accordingly):
      ```
      https://jupyter.pic.es/user/<your_username>/proxy/<port>
      ```

## Inferring on a dataset

Once our SBI pipeline has been trained (irrespective of the specific algorithm considered), it can be used to infer 
on an unseen dataset of generated maps and extract posterior distributions of the corresponding pulsar population 
parameters. The script `pypopsyn/learning/sbi_infer.py` allows us to take an experiment configuration file, a
pretrained model, and a dataset, and run the inference.

### Configuration options

In addition to the testing data loader specified above, we also need to define the directories for saving and loading
inference results using the `save_dir` and `load_dir` parameters under the `infer` field in the configuration file. 
The `load_dir` parameter should point to the location where the `inference.pickle` and `trained_model.pickle` objects
are stored, as these files are required to perform the inference. We must also specify whether to compute the coverage 
probability using the `compute_coverage` flag. If this functionality is disabled, the output will include a corner 
plot `corner_plot_observed_sample.pdf` and samples drawn from the posterior distribution conditioned on the observed
data `samples_posterior.pt` only. If the coverage calculation is enabled, additional outputs will be saved. These 
are as follows:

* `coverage_plot.pdf`: A plot visualizing the coverage probability.
* `coverage_probability.npy`: A matrix of the corresponding coverage probabilities.
* `posterior_samples_test_data.npz`: A NumPy array containing the ground truths and posterior samples for each sample 
   in the test dataset.

Moreover, if `sim_dataset` in the configuration file is set to `True`, the test dataset will be generated in each
round. Otherwise, it is assumed that the directory specified in the `dataset_path` field under `test_dataset_loader` 
contains a pre-generated test dataset for each round.

An example of the inference information looks as follows:

```json
{
    "infer": {
    "sim_dataset": false ,
    "compute_coverage": true,
    "load_dir": "exp_1/learning/models/SBI_ConvolutionMDN/20250403_192124",
    "save_dir": "exp_1/inference"
  }
}
```
!!! note
    The `sbi_infer.py` script requires the folder structure explained above to function correctly and to properly load 
    the test dataset.

Once the inference configuration is set up, we run the inference script by providing the configuration file 
(`--configuration`) as follows:

```commandline
python pypopsyn/learning/sbi_infer.py --configuration tutorials/tutorial_notebooks/config_train_sbi.json 
```

### Inference output

If the inference has been performed successfully, the output of `pypopsyn/learning/sbi_infer.py` will be saved in the 
directory specified in the `save_dir` option. Specifically, inference will create a `logs` folder in this directory 
containing subfolders of the form `name/YYYYMMDD_HHMMSS`, where `YYYYMMDD_HHMMSS` denotes the date and time when the
inference was launched.

The `logs` subdirectories contain the files `profile.json` and `profile.log`, which provide timing and profiling 
information for the inference script executed across all round. Additionally, each `round_{i}` subfolder inside the 
logs directory contains the same files as when testing is enabled while running the `sbi_train.py` script, i.e., we
are producing `corner_plot_observed_sample.pdf`, `coverage_plot.pdf`, `coverage_probability.npy`,
`posterior_samples_test_data.npz`, and `samples_posterior.pt`. For detailed information about each file, see [Training output](#training-output).


