***********************
Learning (sbi) Tutorial
***********************

We use here a simulation-based inference (sbi) framework (see https://sbi-dev.github.io/sbi/ ), a supervised learning approach where training data are suitably labeled and the network has to learn to predict the posterior distribution of the target label associated to the data samples.
The :code:`pypopsyn/learning/train_sbi.py` script allows to train a neural density estimator over a dataset of samples of simulated neutron star populations.
Once the dataset containing the heatmaps or 2D arrays has been created, to train the network over the dataset one can run the script:

::

  python pypopsyn/learning/train.py --configuration pypopsyn/learning/config.json

where the :code:`config.json` file contains all the information needed by the network to train.
This :term:`CLI` can be left unspecified and the script will take the default :code:`pypopsyn/learning/config_sbi.json`.

In this file we can specify various options to configure the training process, e.g., the network model architecture to use as an embedding net, the density estimator type, the input shape of the dataset, and some training hyperparameters.

First of all, we can specify some general settings such as the name of the experiment, the amount of :term:`GPU`\s needed for it, a manual seed for the initialization of the network weights and some profiling options.

::

  "name": "SBI_ConvolutionMDN",
  "n_gpu": 1,
  "set_manual_seed": false,
  "manual_seed": 42,
  "profile_log": "profile.log",
  "profile_json": "profile.json",
  "show_profiling": true,

The first section we need to specify is the architecture of the embedding neural network.
For the sake of the example, we are using a Convolutional Nerual Network which is receiving an array with shape :math:`32 \times 32` with :math:`3` different input channels and is giving a latent vector of size :math:`32` containing a summary of the input feature maps:

::

  "arch": {
    "type": "ModelConvSBI",
    "args": {
        "input_shape": [3, 32, 32],
        "len_output_layer": 32
    }
  },

We also need to specify a scheme to initialize the weights and biases of the network. In this case, we show an example using the :code:`InitializerKaiming`.

::

  {
    "weights_initializer": {
      "type": "InitializerKaiming",
      "args": {}
    },
  }


We can also specify the type of density estimator used to approximate the posterior distribution.
In this example we are setting a mixture density network (mdn) with 10 Gaussian components.

::

  "density_estimator": {
    "type": "mdn",
    "args": {
      "num_components": 10
       }
  },


Next, we need a training loader, responsible for loading the dataset for the training in a representation readable by the network.
Here you have to specify the path to the folder containing the training dataset, the eventual input channels to use in building a multichannel input.
Additionally, we can choose the list of labels from the dataset that we want to consider.
The available input channels and labels are specified in the :code:`train_dataset.csv` file and here they are identified with an index starting from 0.
To select some input channels you need to specify a list containing the indices corresponding to the input channels you would like to consider.
In the example below we are selecting the input channels :code:`survey_PMPS_ppdot_map`, :code:`survey_SMPS_ppdot_map`, :code:`survey_HTRU_ppdot_map` and the labels :code:`B_initial_log10_mean` and :code:`P_initial_log10_mean`.
Furthermore, we can enable on-the-fly normalization or standardization (mutually excluding) for both inputs and labels.
This will use the statistical information contained in the :code:`statistics_train.json` file.
If normalized the input channels will have values in the range between 0 and 1.
If standardized the input channels have values centred around 0 and ranging approximately between -1 and 1.

::

  "training_data_loader": {
    "dataset_path": "data/example_generator_magrot/dataset_train.csv",
    "statistic_path": "data/example_generator_magrot/statistics_train.json",
    "filter_inputs": [9, 10, 11],
    "filter_labels": [12, 13],
    "normalize": false,
    "standardize": true
  },

We need to provide a loader for the test set using the :code:`test_data_loader`.
Such loader must have the same :code:`filter_inputs` and :code:`filter_labels` and the same :code:`normalize` or :code:`standardize`.
In fact, what matters is that both of them are compatible with the network's input shape.

::

  "test_data_loader": {
    "dataset_path": "data/example_generator_magrot/dataset_test.csv",
    "statistic_path": "data/example_generator_magrot/statistics_train.json",
    "filter_inputs": [9, 10, 11],
    "filter_labels": [12, 13],
    "normalize": false,
    "standardize": true
  },

We can set some hyperparameters related to the trainer, i.e. the fraction of the training dataset to use for validation, the training batch size, the initial learning rate for the :code:`Adam` optimizer and the directory path where to save the trained model.

::

   "trainer": {
    "validation_fraction": 0.1,
    "batch_size": 8,
    "lr": 5e-4,
    "save_dir": "data/example_learning_sbi"
  },

Finally we can set up the directory path where the inference results on the test set will be saved.

::

  "infer": {
    "save_dir": "data/example_inference_sbi"
  }

Once you have set up the configuration file, to launch the training script you can simply run the script:

::

  python pypopsyn/learning/train_sbi.py --configuration pypopsyn/learning/config.json

When launching the training script you can also provide some of the parameters contained in the configuration file directly via :term:`CLI`.
For example one can provide the paths to the training, the input channels and the labels to select, the input shape, either to apply normalization or standardization to the input, the batch size, the learning rate value and the path where to save the trained model.
For example you can run a script like the following:

::

  python pypopsyn/learning/train_sbi.py --configuration pypopsyn/learning/config.json --dataset_training data/example_generator_magrot/dataset_train.csv --dataset_statistics data/example_generator_magrot/statistics_train.json --filter_inputs 9 10 11 --filter_labels 12 13 --input_shape 3 32 32 ----len_output_layer 32 --standardize 1 --batch_size 1 --lr 1e-5 --save_dir data/example_learning_sbi


Infer on a Data Set
###################

Once a network has been trained, it can be used to infer on an existing dataset of maps.
To do so the script :code:`pypopsyn/learning/infer_sbi.py` allows you to take an experiment configuration file, a pretrained model, and a data set to run inference.

Once the configuration file is properly set up with the path to the test dataset, to run the inference script you can use a command similar to the following one:

::

    python pypopsyn/learning/infer_sbi.py --configuration pypopsyn/learning/config.json --trained_model data/learning_sbi/models/SBI_ConvolutionMDN/20240606_180938/trained_model.pickle


As for the training script you could provide some arguments via :term:`CLI`, for example the path to the test dataset, the input channels and the labels to select, the input shape, either to apply normalization or standardization to the input:

The inference script will also save the Gaussian coeffiecients for the components of the Gaussian mixture for each of the test sample in :code:`coeff_Gaussians.csv` and the coverage probability diagnostic test resuts in :code:`coverage_plot.pdf` and :code:`coverage_probability.npy`.
You could also specify the argument :code:`--corner_plot True` in order to produce the posterior corner plots for each of the test samples.
