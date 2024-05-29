*****************
Learning Tutorial
*****************

We use here a supervised learning approach where training data are suitably labeled and the network has to learn to predict the target value of the label associated to the data samples.
The :code:`examples/learning/train.py` script allows to train a neural network over a dataset of samples of simulated neutron star populations.
Once the dataset containing the heatmaps or 2D arrays has been created, to train the network over the dataset one can run the script:

.. code-block:: bash

  python examples/learning/train.py --configuration="examples/learning/config.json"

where the :code:`config.json` file contains all the information needed by the network to train. This :term:`CLI` can be left unspecified and the script will take the defautl :code:`examples/learning/config_multiparameter_MLP.json`.

In this file we can specify various options to configure the training process, e.g., the network model architecture to use, the input shape of the dataset, or the number of output parameters to predict.

First of all, we can specify some general settings such as the name of the experiment, the amount of :term:`GPU`\s needed for it, the amount of trials to perform if convergence is not reached, and the specific thresholds for convergence for each one of the predicted or output parameters.
The script will try to perform several training trials until all convergence thresholds are met or the number of trials is reached.
If convergence is not reached in the number of trials indicated the best trained model is saved anyway.

.. code-block:: json

  {
    "name": "Linear",
    "trials": 8,
    "convergence": {
      "h_c_threshold": 0.5,
      "vk_c_threshold": 10.0,
      "sigma_k_threshold": 10.0
    },
    "n_gpu": 0,
  }

The first section we need to specify is the architecture. For the sake of the example, we are using a linear network with fully connected layers which is receiving an array with shape :math:`64 \times 64` with :math:`4` different input channels and is giving as output the predicted value of one parameter for each sample:

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

Next, we need a training loader, responsible for loading the dataset for the training in a representation readable by the network.
Here you have to specify the path to the folder containing the training dataset, the batch size, and the eventual input channels to use in building a multichannel input.
Additionally, we can choose the list of labels from the dataset that we want to consider.
The available input channels and labels are specified in the :code:`train_dataset.csv` file and here they are identified with an index starting from 0.
To select some input channels you need to specify a list containing the indices corresponding to the input channels you would like to consider.
In the example below we are selecting the input channels :code:`position_map_xy`, :code:`velocity_map_xy_vr`, :code:`velocity_map_xy_vphi` and :code:`velocity_map_xy_vz` and the label :code:`h_c`.
Furthermore, we can enable on-the-fly normalization or standardization (mutually excluding) for both inputs and labels.
This will use the statistical information contained in the :code:`statistics_train.json` file.
If normalized the input channels will have values in the range between 0 and 1.
If standardized the input channels have values centred around 0 and ranging approximately between -1 and 1.

.. code-block:: json

  {
    "training_data_loader": {
      "type": "LoaderMultichannelArray",
      "args": {
        "dataset_path": "generated_dataset/dataset_train.csv",
        "statistic_path": "generated_dataset/statistics_train.json",
        "batch_size": 8,
        "num_workers": 1,
        "filter_inputs": [0, 3, 4, 5],
        "filter_labels": [14],
        "shuffle": true,
        "normalize": false,
        "standardize": false
      }
    },
  }

We need to provide a loader for the validation set using the :code:`validation_data_loader`.
Such loader must have the same :code:`filter_inputs` and :code:`filter_labels` and be of the same :code:`type` as the training data loader.
In fact, what matters is that both of them are compatible with the network's input shape.

.. code-block:: json

  {
    "validation_data_loader": {
      "type": "LoaderMultichannelArray",
      "args": {
        "dataset_path": "generated_dataset/dataset_valid.csv",
        "statistic_path": "generated_dataset/statistics_train.json",
        "batch_size": 8,
        "num_workers": 1,
        "filter_inputs": [0, 3, 4, 5],
        "filter_labels": [14],
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

We have to specify the loss function to minimize and the metric to monitor the predictive accuracy of the neural network model over the validation set during training. The implemeted loss function :code:`LossRMSE` evaluates the :term:`RMSE` between the output of the network and the target labels over every training epoch. For the accuracy metric a similiar :term:`RMSE` metric is implemented called :code:`MetriAccuracyRMSE`. An optimal value of the :term:`RMSE` should be around 0 for a well-trained network:

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

We can also set up a scheduler for the learning rate, which can update the value of the learning rate after a number of epochs specified by the :code:`step_size` parameter, by multiplying it by a factor :code:`gamma`.
In this example after :math:`128` training epochs the learning rate is multiplied by a factor :math:`0.1`.
Note that scheduling has different effects depending on the optimizer.
For example for an adaptive optimizer like ADAM the learning rate is automatically adjusted during training, depending on the values of the loss gradients with respect to the network weights.
Therefore the learning scheduler could not be effective in this case.
On the other hand, for optimizer where the learning rate is fixed, rescheduling its value after some training epochs could help to converge faster towards a minimum of the loss landscape.

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

Once you have set up the configuration file, to launch the training script you can simply run the script:

.. code-block:: bash

  python examples/learning/train.py --configuration examples/learning/config.json

When launching the training script you can also provide some of the parameters contained in the configuration file directly via :term:`CLI`.
For example one can provide the paths to the training and validation dataset, the input channels and the labels to ignore, the input shape, the number of parameters to predict, either to apply normalization or standardization to the input, the batch size, the learning rate value and the path where to save the trained model.
For example you can run a script like the following:

.. code-block:: bash

  python examples/learning/train.py --configuration examples/learning/config.json --dataset_training generated_dataset/dataset_train.csv --dataset_validation generated_dataset/dataset_valid.csv --dataset_statistics generated_dataset/statistics_train.json --filter_inputs 0 3 4 5 --filter_labels 14 --input_shape 4 64 64 --num_parameters 1 --normalize 1 --batch_size 1 --lr 1e-5 --save_dir training_results

The :code:`examples/experiment_launcher.py` script allows you to specify a list of experiment commands in a text file like:

.. code-block:: bash

  python examples/learning/train.py --dataset_training generated_dataset/array_64/train_dataset.csv --dataset_validation generated_dataset/array_64/dataset_valid.csv --dataset_statistics generated_dataset/array_64/statistics_train.json --input_shape 4 64 64 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r64_gc_position_velocity
  python examples/learning/train.py --dataset_training generated_dataset/array_128/train_dataset.csv --dataset_validation generated_dataset/array_128/dataset_valid.csv --dataset_statistics generated_dataset/array_128/statistics_train.json --input_shape 4 128 128 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r128_gc_position_velocity
  python examples/learning/train.py --dataset_training generated_dataset/array_256/train_dataset.csv --dataset_validation generated_dataset/array_256/dataset_valid.csv --dataset_statistics generated_dataset/array_256/statistics_train.json --input_shape 4 256 256 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r256_gc_position_velocity
  python examples/learning/train.py --dataset_training generated_dataset/array_512/train_dataset.csv --dataset_validation generated_dataset/array_512/dataset_valid.csv --dataset_statistics generated_dataset/array_512/statistics_train.json --input_shape 4 512 512 --lr 1e-8 --filter_inputs 1 2 6 7 --batch_size 1 --save_dir learning_results/s8_r512_gc_position_velocity

By default, the command list will be held in :code:`examples/command_list.txt`. Each line should contain one full command (including the :code:`python` program call) to execute an experiment. The script will execute those experiments automatically and in parallel providing a number of maximum simultaneous :code:`--processes`.
Obviously, this number of processes should be set as a function of the number of available threads/cores.

A custom experiments file can be specified with the :code:`--command_list` parameter:

.. code-block:: bash

  python examples/experiment_launcher.py --command_list examples/learning/experiment_list.txt --processes 2

This combined with an intelligent use of the :code:`--save_dir` :term:`CLI` argument will let you run many experiments unattended and check them asynchronously.

Infer on a Data Set
###################

Once a network has been trained, it can be used to infer on an existing dataset of density and velocity maps.
To do so the script :code:`examples/learning/infer.py` allows you to take an experiment configuration file, a pretrained model, and a data set to run inference on selected samples for that dataset.

To use this inference script you will need to provide a dataset to infer (:code:`--dataset`), a checkpoint with a pretrained model (:code:`--weights`), the configuration file of the experiment that generated such model (:code:`--c`) and a directory path where to save the CSV file with the inference results. For instance:

.. code-block:: bash

    python examples/learning/infer.py --c examples/learning/config_multiparameter_MLP.json --dataset examples/data/8_samples/dataset.csv --resume examples/learning/saved/models/Linear/0407_175854/model_best.pth --save_dir inference_results

Then you can use the :code:`--samples` argument to provide a list of samples you would like to infer (their indices in the dataset) or just leave it blank to infer over all.
Make sure that the data set used for inference has the same input configuration as the data set used for training the model, i.e., same input shape, number of labels to predict, normalization etc..
If the inference dataset does not match an error is raised automatically by pytorch.
For example if you put the wrong resolution for the input maps the error raised is similar to:

.. code-block:: bash

    RuntimeError: Error(s) in loading state_dict for ModelConv:
        size mismatch for fc1.weight: copying a param with shape torch.Size([64, 26880]) from checkpoint, the shape in current model is torch.Size([64, 12544]).

If the input channels do not match, an error like the following is raised:

.. code-block:: bash

    ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 128 and the array at index 1 has size 64

The output will be the labels (ground truth) for each sample and the corresponding prediction printed on terminal and saved in a CSV file :code:`inference_results.csv`.
For example, in case of inference over the two parameters :code:`h_c` and :code:`sigma_k` the output file would be like this:

.. code-block:: bash

    target:h_c,target:sigma_k,predicted:h_c,predicted:sigma_k
    1.594645619392395,204.6456756591797,1.600591778755188,208.88429260253906
    1.9688189029693604,127.5905532836914,1.961457371711731,124.93790435791016
    1.017795443534851,452.32281494140625,1.0061225891113281,449.244873046875
    0.16031496226787567,446.8188781738281,0.160542294383049,438.3158874511719
    1.6414172649383545,237.66929626464844,1.647267460823059,241.9035186767578
    1.6258267164230347,589.9212646484375,1.600623369216919,595.2665405273438
    1.1892913579940796,111.07874298095703,1.1856337785720825,110.28069305419922
    0.8930708765983582,441.3149719238281,0.8915233612060547,433.6885986328125
    1.2672441005706787,17.511810302734375,1.3246393203735352,11.903473854064941
    0.08236220479011536,122.08661651611328,0.10178111493587494,130.51028442382812
    2.0,485.3464660644531,2.0088584423065186,490.4414367675781
    0.20708660781383514,325.7322692871094,0.2022785246372223,313.9435729980469
    1.2204724550247192,611.93701171875,1.153143048286438,622.7503662109375
    1.0957480669021606,254.1811065673828,1.0770119428634644,255.84629821777344
    1.5634645223617554,490.85040283203125,1.507364273071289,493.6461181640625
    ...

To plot the inference results in the form of residuals plot you can use the jupyter notebooks :code:`inference_results_1par_plots.ipynb` or :code:`inference_results_2par_plots.ipynb` for the single parameter or the two parameter inference respectively.
To run the first script notebook, you need to specify the path to the :code:`inference_results.csv` files for either one or both the :code:`h_c` and :code:`sigma_k` parameters directly in the notebook.
To run the second notebook, you need to provide the path to the :code:`inference_results.csv` containing the prediction on both parameters directly in the notebook.