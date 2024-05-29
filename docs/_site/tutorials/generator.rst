******************
Generator Tutorial
******************


Once a set of simulated populations is created by running one of the simulator scripts (see above for more information), it is possible to generate a dataset of synthetic representations of the simulations that is readable by a machine-learning pipeline.
Depending on the type of simulations that have been performed, two types of generator scripts can be used, the :code:`examples/generator/generate_dataset.py` or the :code:`examples/generator/generate_dataset_survey.py`.

The first script :code:`examples/generator/generate_dataset.py` can be used if simulations have been run using the :code:`simulate_population_full` script, since it will read the corresponding :code:`final_population.pkl.gz` output files.
For each simulated population, this script can generate a set of density maps in the form of either :code:`.png` images or 2D numpy :code:`.npy` arrays.
These maps store the spatial density and velocity information in galactocentric or equatorial (ICRS) reference frames and the density in a :math:`P-\dot{P}` diagram of all the evolved neutron stars.

The second script :code:`examples/generator/generate_dataset_survey.py` can be used if simulations have been run using the :code:`simulate_population_magrot_det` script, since it will read the corresponding :code:`.pkl.gz` output files that are produced for each of the simulated surveys.
For each simulated population, this script can generate a set of density maps in the form of either :code:`.png` images or 2D numpy :code:`.npy` arrays.
These maps store the spatial density and proper motion information in equatorial (ICRS) reference frames and the density in a :math:`P-\dot{P}` diagram of the simulated neutron stars that have been detected by the modelled surveys only.

Suppose that you have created a set of simulated populations stored in :code:`simulated_data` using the :code:`simulate_population_full` script.
If you want to create a dataset of 2D arrays storing the spatial density and the velocity information of the simulated neutron stars with a resolution of :math:`64 \times 64` and the density in the :math:`P-\dot{P}` diagram with a resolution of :math:`32 \times 32` you can run the following command:

.. code-block:: bash

  python examples/generator/generate_dataset.py --data simulated_data --save_dir generated_dataset --data_type array --resolution_dyn 64 --resolution_ppdot 32

You need to provide the path to the location where the simulated populations data are stored, the path where to save the dataset that you are going to create, the type of the representations (you can choose :code:`array` or :code:`image`) and the resolution.
By running the script, the folder :code:`generated_dataset` is created where a set of 2D arrays are stored for each simulated population (sample) along with a :code:`dataset_full.csv` file containing all the information about the dataset and a :code:`statistics_full.json` file containing the statistical information on each label.
The CSV file provides one line for each sample in the dataset in which we indicate the file path for its 2D arrays (potential input channels for the machine learning pipeline) and the values for its parameters (labels) like this:

::

  input:position_map_xy,input:position_map_xz,input:position_map_radec,input:velocity_map_xy_vr,input:velocity_map_xy_vphi,input:velocity_map_xy_vz,input:velocity_map_vra,input:velocity_map_vdec,input:ppdot_map,B_initial_log10_mean,B_initial_log10_sigma,P_initial_mean,P_initial_sigma,h_c,kick_model,sigma_k
  generated_dataset/position_map_xy_0.npy,generated_dataset/position_map_xz_0.npy,generated_dataset/position_map_radec_0.npy,generated_dataset/velocity_map_xy_vr_0.npy,generated_dataset/velocity_map_xy_vphi_0.npy,generated_dataset/velocity_map_xy_vz_0.npy,generated_dataset/velocity_map_vra_0.npy,generated_dataset/velocity_map_vdec_0.npy,generated_dataset/ppdot_map_0.png,13.2,0.62,0.22,0.42,0.18,km_maxwell,100
  generated_dataset/position_map_xy_1.npy,generated_dataset/position_map_xz_1.npy,generated_dataset/position_map_radec_1.npy,generated_dataset/velocity_map_xy_vr_1.npy,generated_dataset/velocity_map_xy_vphi_1.npy,generated_dataset/velocity_map_xy_vz_1.npy,generated_dataset/velocity_map_vra_1.npy,generated_dataset/velocity_map_vdec_1.npy,generated_dataset/ppdot_map_0.png,13.2,0.62,0.22,0.42,0.18,km_maxwell,200
  generated_dataset/position_map_xy_2.npy,generated_dataset/position_map_xz_2.npy,generated_dataset/position_map_radec_2.npy,generated_dataset/velocity_map_xy_vr_2.npy,generated_dataset/velocity_map_xy_vphi_2.npy,generated_dataset/velocity_map_xy_vz_2.npy,generated_dataset/velocity_map_vra_2.npy,generated_dataset/velocity_map_vdec_2.npy,generated_dataset/ppdot_map_0.png,13.2,0.62,0.22,0.42,0.18,km_maxwell,300
  generated_dataset/position_map_xy_3.npy,generated_dataset/position_map_xz_3.npy,generated_dataset/position_map_radec_3.npy,generated_dataset/velocity_map_xy_vr_3.npy,generated_dataset/velocity_map_xy_vphi_3.npy,generated_dataset/velocity_map_xy_vz_3.npy,generated_dataset/velocity_map_vra_3.npy,generated_dataset/velocity_map_vdec_3.npy,generated_dataset/ppdot_map_0.png,13.2,0.62,0.22,0.42,0.18,km_maxwell,400

In this way the format is easily compatible with our machine-learning pipeline.

The JSON file contains information about the mean, standard deviation, minimum and maximum values for each of the dataset labels.
This statistical information will be used during the training process if one wants to normalize or standardize the label values.

You can also choose to split the dataset into training/validation or into training/validation/test sets.
To do this you can run the :code:`dataset_splitter.py` script in the :code:`scripts` folder.
To generate a dataset split into two subsets, one specifically for training and the other for validation, you can specify a fraction of the total dataset that will form the validation subset by passing the argument :code:`valid_split` in the :code:`dataset_splitter` script. For example:

.. code-block:: bash

 python scripts/dataset_splitter.py --dataset_path generated_dataset --valid_split 0.2

This will create two files :code:`dataset_train.csv` and :code:`dataset_valid.csv` that will specify the samples belonging to the train dataset (80 % of the total dataset in this case) and the ones belonging to the validation dataset  (20 % of the total dataset).
The split is performed by randomly sampling the validation subset from the total dataset according to the specified split fraction.
In this case, the :code:`statistics_train.json` file will contain the statistics computed on the labels of the training set only.

If you also want to create a test set in addition to the training and validation sets, you can specify the argument :code:`test_split`, which sets the fraction of the total dataset to be dedicated for testing purposes.
In this case, the :code:`valid_split` argument will specify the fraction of the dataset not used for testing but instead dedicated for validation.

.. code-block:: bash

 python scripts/dataset_splitter.py --dataset_path generated_dataset --test_split 0.1 --valid_split 0.2

This will create three files :code:`dataset_train.csv`, :code:`dataset_valid.csv` and :code:`dataset_test.csv` that will specify the samples belonging to the train dataset (80 % of the dataset not used for testing in this case), the ones belonging to the validation dataset  (20 % of the dataset not used for testing) and the ones belonging to the test set (10 % of the total dataset), respectively.
Again the split is performed by randomly sampling the test and validation subsets from the dataset according to the specified split fractions.
In this case, the :code:`statistics_train.json` file will also contain the statistics computed on the labels of the training set only.

*******************
Experiment Launcher
*******************

The :code:`examples/experiment_launcher.py` script allows you to specify a list of experiment commands in a text file like:

.. code-block:: bash

  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_64 --type array --resolution_dyn 64 --resolution_ppdot 32
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_128 --type array --resolution_dyn 128 --resolution_ppdot 32
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_256 --type array --resolution_dyn 256 --resolution_ppdot 32
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_512 --type array --resolution_dyn 512 --resolution_ppdot 32

By default, the command list will be held in :code:`examples/command_list.txt`. Each line should contain one full command (including the :code:`python` program call) to execute an experiment. The script will execute those experiments automatically and in parallel providing a number of maximum simultaneous :code:`--processes`.
Obviously, this number of processes should be set as a function of the number of available threads/cores.

A custom experiments file can be specified with the :code:`--command_list` parameter:

.. code-block:: bash

  python examples/experiment_launcher.py --command_list examples/generator/experiment_list.txt --processes 2

This combined with an intelligent use of the :code:`--save_dir` :term:`CLI` argument will let you run many experiments unattended and check them asynchronously.