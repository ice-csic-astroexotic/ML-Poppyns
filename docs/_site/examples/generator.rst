*****************
Generator Example
*****************

Once a set of simulated populations is created by running the simulator script example (see above for more information about it), now you would like to generate a dataset readable by a machine learning pipeline. To do that you can run the :code:`examples/generator/generate_dataset.py`. This script can generate either a heatmap/density maps :code:`.png` images dataset or a 2D numpy :code:`.npy` arrays dataset storing the data of 2D histograms. For example the heatmaps (or 2D arrays) can represent the spatial density and velocity maps of the simulated neutron stars in the Galaxy.  

Suppose that you have created a set of simulated populations stored in :code:`multirun/2020-04-16/15-45-46`. If you want to create a dataset of 2D arrays storing the spatial density and the velocity maps of the simulated neutron stars with a resolution of :math:`64 \times 64` you can run the script: 

.. code-block:: bash

  python examples/generator/generate_dataset.py --data_path multirun/2020-04-16/15-45-46 --dataset_name array_dataset --type array --resolution 64

you need to provide the path to the location where the simulated populations are stored, the name of the dataset that you are going to create, the type of the representations (you can choose :code:`array` or :code:`image`) and the resolution. By running the script, the folder :code:`examples/data/array_dataset` is created where a set of 2D arrays are stored for each sample along with a :code:`dataset.csv` file containing all the information about the dataset. Such CSV file provides one line for each sample in the dataset in which we indicate the file path for its 2D arrays and the values for its parameters like this:

::

  input:position_map_xy,input:position_map_xz,input:velocity_map_xy_vr,input:velocity_map_xy_vphi,input:velocity_map_xy_vz,vk_c
  examples/data/2020-04-16/15-45-46/array_dataset/position_map_xy_pop_0.npy,examples/data/2020-04-16/15-45-46/array_dataset/position_map_xz_pop_0.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vr_pop_0.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vphi_pop_0.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vz_pop_0.npy,100.0
  examples/data/2020-04-16/15-45-46/array_dataset/position_map_xy_pop_1.npy,examples/data/2020-04-16/15-45-46/array_dataset/position_map_xz_pop_1.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vr_pop_1.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vphi_pop_1.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vz_pop_1.npy,200.0
  examples/data/2020-04-16/15-45-46/array_dataset/position_map_xy_pop_2.npy,examples/data/2020-04-16/15-45-46/array_dataset/position_map_xz_pop_2.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vr_pop_2.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vphi_pop_2.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vz_pop_2.npy,300.0
  examples/data/2020-04-16/15-45-46/array_dataset/position_map_xy_pop_3.npy,examples/data/2020-04-16/15-45-46/array_dataset/position_map_xz_pop_3.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vr_pop_3.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vphi_pop_3.npy,examples/data/2020-04-16/15-45-46/array_dataset/velocity_map_xy_vz_pop_3.npy,600.0

In this way it is easy for later loading it into the learning subpackage.

.. note::

  If you don't want to generate all the samples in the population, you can use the :code:`--samples {int}` argument to specify a number of samples to select from the population. They will be equally spaced. If no samples are indicated, the whole population will be taken into account. NOTE: This will be hard to do when we have multiple parameters varying in the population.