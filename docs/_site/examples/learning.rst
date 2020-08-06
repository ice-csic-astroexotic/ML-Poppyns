****************
Learning Example
****************

The :code:`examples/learning/train.py` script allows to train a neural network over a dataset of samples of simulated neutron stars populations. Once the dataset containing the heatmaps or 2D arrays has been created, to train the network over the dataset one can run the script:

.. code-block:: bash

  python examples/learning/train.py --configuration="examples/learning/config.json"

where the :code:`config.json` file contains all the information needed by the network to train. This :term:`CLI` can be left unspecified and the script will take the defautl :code:`examples/learning/config_multiparameter.json`.

In this file we can specify various options to configure the training process, e.g., the network model architecture to use, the input shape of the dataset, or the number of output parameters to predict.

First of all, we can specify some general settings such as the name of the experiment, the amount of :term:`GPU`s needed for it, the amount of trials to perform if convergence is not reached, and the specific thresholds for convergence for each one of the predicted or output parameters. The script will try to perform several training trials until all convergence thresholds are met or the number of trials is reached.

.. code-block:: json

  {
    "name": "Linear",
    "trials": 8,
    "convergence": {
      "h_mean": 0.01,
      "vp_mean": 5.0
    },
    "n_gpu": 0,
  }

The first section we need to specify is the architecture. For the sake of the example, we are using a linear network with fully connected layers which is receiving an array with shape :math:`64 \times 64` with :math:`4` different input channels and is giving as output the predicted value of the parameter for each sample:

.. code-block:: json

  {
    "arch": {
      "type": "ModelLinear",
      "args": {
          "input_shape": [4, 64, 64],
          "num_parameters": 1
      }
    },
  }

We also need to specify a scheme to initialize the weights and biases of the network. In this case, we show an example using the :code:`InitializerKaiming`.

.. code-block:: json

  {
    "weights_initializer": {
      "type": "InitializerKaiming",
      "args": {}
    },
  }

Next, we need a training loader, responsible for loading the dataset for the training in a representation readable by the network. Here you have to specify the path to the folder containing the dataset, the batch size, and the eventual input channels to ignore in building a multichannel input by specifying a list of index starting from 0. Additionally, we can choose to ignore a list of labels from the dataset. Furthermore, we can (mutually excluding) to enable on-the-fly normalization or standardization for both inputs and labels.

.. code-block:: json

  {
    "data_loader": {
      "type": "LoaderMultichannelArray",
      "args": {
        "data_path": "examples/data/multiparameter/dataset.csv",
        "batch_size": 8,
        "num_workers": 1,
        "ignored_inputs": [1, 2, 6, 7],
        "ignored_labels": [],
        "normalize": false,
        "standardize": false
      }
    },
  }

We can also optionally provide a loader for the validation set using the :code:`validation_data_loader`. Such loader must have the same :code:`ignored_inputs` and :code:`ignored_labels` and be of the same :code:`type` as the training data loader. In fact, what matters is that both of them are compatible with the network's input shape.

.. code-block:: json

  {
    "validation_data_loader": {
      "type": "LoaderMultichannelArray",
      "args": {
        "data_path": "examples/data/multiparameter/dataset.csv",
        "batch_size": 8,
        "num_workers": 1,
        "ignored_inputs": [1, 2, 6, 7],
        "ignored_labels": [],
        "shuffle": false,
        "normalize": false,
        "standardize": false
      }
    },
  }

We can set the optimizer type, which regulates the training process (see `here <https://pytorch.org/docs/stable/optim.html>`_ for the different type of PyTorch optimizers). Each specific optimizer has a set of extra parameters that can be provided (such as :code:`lr` or :code:`weight_decay` for ADAM).

.. code-block:: json

  {
    "optimizer": {
      "type": "Adam",
      "args":{
        "lr": 1e-5,
        "weight_decay": 0.0
      }
    },
  }

The loss function to use and the metric to monitor the accuracy of the neural network model during training. The implemeted loss function :code:`LossRMSE` evaluates the average :term:`RMSE` between the output of the network and the target labels over every batch. For the accuracy metric a root mean squared error metric is implemented called :code:`MetriAccuracyRMSE`. An optimal value of the :term:`RMSE` should be around 0 for a well-trained network:

.. code-block:: json

  {
    "loss": {
        "type": "LossRMSE",
        "args": {}
    },

    "metric": {
        "type": "MetricAccuracyRMSE",
        "args": {}
    },
  }

A scheduler for the learning rate, which can update the value of the learning rate after a number of epochs specified by the :code:`step_size` parameter, by multiplying it by a factor :code:`gamma`. In this example after :math:`200` training epochs the learning rate is multiplied by a factor :math:`0.1`. Note that scheduling has different effects depending on the optimizer.

.. code-block:: json

  {
    "lr_scheduler": {
      "type": "StepLR",
      "args": {
        "step_size": 128,
        "gamma": 0.1
      }
    },
  }

Some parameters for the training routine such as the total number of epochs, the directory path where to save the best model network and some checkpoints during the training process.

.. code-block:: JSON

  {
    "trainer": {
      "epochs": 1024,
      "save_dir": "examples/learning/saved",
      "save_period": 1000,
      "verbosity": 1,
      "early_stop": 32,
      "tensorboard": true
    }
  }

When launching the training script you can also choose to normalize or standardize the input channels. If normalized the input channels will have values in the range between 0 and 1. If standardized the input channels have values centred around 0 and ranging approximately between -1 and 1. To normalize/standardize the input channels during training you can run either one or the other of the following scripts:

.. code-block:: bash

  python examples/learning/train.py --configuration="examples/learning/config.json" --normalize 1
  python examples/learning/train.py --configuration="examples/learning/config.json" --standardize 1

The :code:`examples/experiment_launcher.py` script allows you to specify a list of experiment commands in a text file like:

.. code-block:: bash

  python examples/learning/train.py --dataset examples/data/8_array_64/dataset.csv --input_shape 4 64 64 --lr 1e-8 --ignored_inputs 1 --batch_size 1 --save_dir examples/learning/saved/s8_r64_position_velocity
  python examples/learning/train.py --dataset examples/data/8_array_128/dataset.csv --input_shape 4 128 128 --lr 1e-8 --ignored_inputs 1 --batch_size 1 --save_dir examples/learning/saved/s8_r128_position_velocity
  python examples/learning/train.py --dataset examples/data/8_array_256/dataset.csv --input_shape 4 256 256 --lr 1e-8 --ignored_inputs 1 --batch_size 1 --save_dir examples/learning/saved/s8_r256_position_velocity
  python examples/learning/train.py --dataset examples/data/8_array_512/dataset.csv --input_shape 4 512 512 --lr 1e-8 --ignored_inputs 1 --batch_size 1 --save_dir examples/learning/saved/s8_r512_position_velocity

by default, the command list will be held in :code:`examples/command_list.txt`. Each line should contain one full command (including the :code:`python` program call) to execute an experiment. The script will execute those experiments automatically and in parallel providing a number of maximum simultaneous :code:`--processes`:

.. code-block:: bash

  python examples/experiment_launcher.py --processes 2

Obviously, this number of processes should be set as a function of the number of available threads/cores.

A custom experiments file can be specified with the :code:`--command_list` parameter:

.. code-block:: bash

  python examples/experiment_launcher.py --command_list examples/learning/custom.txt --processes 2

This combined with an intelligent use of the :code:`--save_dir` :term:`CLI` argument will let you run many experiments unattended and check them asynchronously.