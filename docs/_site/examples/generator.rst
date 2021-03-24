*****************
Generator Example
*****************

Once a set of simulated populations is created by running the simulator script example (see above for more information about it), now you would like to generate a dataset readable by a machine learning pipeline.
To do that you can run the :code:`examples/generator/generate_dataset.py`.
This script can generate either a heatmap/density maps :code:`.png` images dataset or a 2D numpy :code:`.npy` arrays dataset storing the data of 2D histograms.
For example the heatmaps (or 2D arrays) can represent the spatial density and velocity information of the simulated neutron stars in the Galaxy in galactocentric or equatorial (ICRS) reference frames.

Suppose that you have created a set of simulated populations stored in :code:`simulated_data`.
If you want to create a dataset of 2D arrays storing the spatial density and the velocity information of the simulated neutron stars with a resolution of :math:`64 \times 64` you can run the script:

.. code-block:: bash

  python examples/generator/generate_dataset.py --data simulated_data --save_dir generated_dataset --type array --resolution 64

You need to provide the path to the location where the simulated populations data are stored, the path where to save the dataset that you are going to create, the type of the representations (you can choose :code:`array` or :code:`image`) and the resolution.
By running the script, the folder :code:`generated_dataset` is created where a set of 2D arrays are stored for each simulated population (sample) along with a :code:`dataset.csv` file containing all the information about the dataset.
Such CSV file provides one line for each sample in the dataset in which we indicate the file path for its 2D arrays (potential input channels for the machine learning pipeline) and the values for its parameters (labels) like this:

::

  input:position_map_xy,input:position_map_xz,input:position_map_radec,input:velocity_map_xy_vr,input:velocity_map_xy_vphi,input:velocity_map_xy_vz,input:velocity_map_vra,input:velocity_map_vdec,h_c,sigma_k
  generated_dataset/position_map_xy_0.npy,generated_dataset/position_map_xz_0.npy,generated_dataset/position_map_radec_0.npy,generated_dataset/velocity_map_xy_vr_0.npy,generated_dataset/velocity_map_xy_vphi_0.npy,generated_dataset/velocity_map_xy_vz_0.npy,generated_dataset/velocity_map_vra_0.npy,generated_dataset/velocity_map_vdec_0.npy,0.18,100
  generated_dataset/position_map_xy_1.npy,generated_dataset/position_map_xz_1.npy,generated_dataset/position_map_radec_1.npy,generated_dataset/velocity_map_xy_vr_1.npy,generated_dataset/velocity_map_xy_vphi_1.npy,generated_dataset/velocity_map_xy_vz_1.npy,generated_dataset/velocity_map_vra_1.npy,generated_dataset/velocity_map_vdec_1.npy,0.18,200
  generated_dataset/position_map_xy_2.npy,generated_dataset/position_map_xz_2.npy,generated_dataset/position_map_radec_2.npy,generated_dataset/velocity_map_xy_vr_2.npy,generated_dataset/velocity_map_xy_vphi_2.npy,generated_dataset/velocity_map_xy_vz_2.npy,generated_dataset/velocity_map_vra_2.npy,generated_dataset/velocity_map_vdec_2.npy,0.18,300
  generated_dataset/position_map_xy_3.npy,generated_dataset/position_map_xz_3.npy,generated_dataset/position_map_radec_3.npy,generated_dataset/velocity_map_xy_vr_3.npy,generated_dataset/velocity_map_xy_vphi_3.npy,generated_dataset/velocity_map_xy_vz_3.npy,generated_dataset/velocity_map_vra_3.npy,generated_dataset/velocity_map_vdec_3.npy,0.18,400

In this way the format is easily compatible with our machine-learning pipeline.

To generate a dataset split into two subsets one specific for training and the other for validation you can specify a fraction of the total dataset that will form the validation subset by passing the argument :code:`split` in the generator script. For example:

.. code-block:: bash

 python examples/generator/generate_dataset.py --split 0.2 --data simulated_data --save_dir generated_dataset --type array --resolution 128

This will create a dataset described in the file :code:`dataset.csv` along with two other files :code:`train_dataset.csv` and :code:`valid_dataset.csv` that will specify the samples belonging to the train dataset (80 % of the total dataset in this case) and the ones belonging to the validation dataset  (20 % of the total dataset).
The split is performed by randomly sampling the validation subset from the total dataset according to the specified split fraction.

The :code:`examples/experiment_launcher.py` script allows you to specify a list of experiment commands in a text file like:

.. code-block:: bash

  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_64 --type array --resolution 64
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_128 --type array --resolution 128
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_256 --type array --resolution 256
  python  examples/generator/generate_dataset.py --data simulated_data --save_dir generated_data/array_512 --type array --resolution 512

By default, the command list will be held in :code:`examples/command_list.txt`. Each line should contain one full command (including the :code:`python` program call) to execute an experiment. The script will execute those experiments automatically and in parallel providing a number of maximum simultaneous :code:`--processes`.
Obviously, this number of processes should be set as a function of the number of available threads/cores.

A custom experiments file can be specified with the :code:`--command_list` parameter:

.. code-block:: bash

  python examples/experiment_launcher.py --command_list examples/generator/experiment_list.txt --processes 2

This combined with an intelligent use of the :code:`--save_dir` :term:`CLI` argument will let you run many experiments unattended and check them asynchronously.