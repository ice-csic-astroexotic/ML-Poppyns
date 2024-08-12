# Generating density maps

Once a set of simulated populations is created by running one of the simulator scripts (see the simulator tutorial), it is possible to generate a dataset of synthetic representations of the simulations that is readable by a machine-learning pipeline.
Depending on the type of simulations that have been performed, two types of generator scripts can be used, the `pypopsyn/generator/generate_dataset_full.py` or the `pypopsyn/generator/generate_dataset_survey.py`.

The first script `pypopsyn/generator/generate_dataset_full.py` can be used if simulations have been run using the `simulate_population_full` script, since it will read the corresponding `final_population.pkl.gz` output files.
For each simulated population, this script can generate a set of density maps in the form of either `.png` images or 2D numpy `.npy` arrays.
These maps store the spatial density and velocity information in galactocentric or equatorial (ICRS) reference frames and the density in a $P-\dot{P}$ diagram of all the evolved neutron stars.

The second script `pypopsyn/generator/generate_dataset_surveys.py` can be used if simulations have been run using the `simulate_population_magrot_det` script, since it will read the corresponding `.pkl.gz` output files that are produced for each of the simulated surveys.
For each simulated population, this script can generate a set of density maps in the form of either `.png` images or 2D numpy `.npy` arrays.
These maps store the spatial density and proper motion information in equatorial (ICRS) reference frames and the density in a $P-\dot{P}$ diagram of the simulated neutron stars that have been detected by the modelled surveys only.

Suppose that you have created a set of simulated populations stored in `data/example_simulation_helper_magrot` using the `simulate_population_magrot_det.py` script.
If you want to create a dataset of 2D arrays storing the spatial density and the velocity information of the simulated neutron stars with a resolution of $32 \times 32$ and the density and radio flux information in the $P-\dot{P}$ diagram with a resolution of $32 \times 32$ you can run the following command:
```commandline
python pypopsyn/generator/generate_dataset_surveys.py --data data/example_simulation_helper_magrot --save_dir data/example_generator_magrot --data_type array --resolution_dyn 32 --resolution_ppdot 32
```
You need to provide the path to the location where the simulated populations data are stored, the path where to save the dataset that you are going to create, the type of the representations (you can choose `array` or `image`) and the resolution.
By running the script, the folder `data/example_generator_magrot` is created where a set of 2D arrays are stored for each simulated population (sample) along with a `dataset_full.csv` file containing all the information about the dataset and a `statistics_full.json` file containing the statistical information on each label.
The CSV file provides one line for each sample in the dataset in which we indicate the file path for its 2D arrays (potential input channels for the machine learning pipeline) and the values for its parameters (labels) like this:
```commandline
input:survey_PMPS_position_map_radec,input:survey_SMPS_position_map_radec,input:survey_HTRU_position_map_radec,input:survey_PMPS_velocity_map_vra,input:survey_SMPS_velocity_map_vra,input:survey_HTRU_velocity_map_vra,input:survey_PMPS_velocity_map_vdec,input:survey_SMPS_velocity_map_vdec,input:survey_HTRU_velocity_map_vdec,input:survey_PMPS_ppdot_map,input:survey_SMPS_ppdot_map,input:survey_HTRU_ppdot_map,input:survey_PMPS_ppdot_map_fluxes,input:survey_SMPS_ppdot_map_fluxes,input:survey_HTRU_ppdot_map_fluxes,B_initial_log10_mean,P_initial_log10_mean
data/example_generator_magrot/survey_PMPS_position_map_radec_0.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_0.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_0.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_0.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_0.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_0.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_0.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_0.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_0.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_0.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_0.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_0.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_0.npy,12.586280501149703,-1.3096012845994798
data/example_generator_magrot/survey_PMPS_position_map_radec_1.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_1.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_1.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_1.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_1.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_1.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_1.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_1.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_1.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_1.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_1.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_1.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_1.npy,12.217080081747753,-0.5284096562094787
data/example_generator_magrot/survey_PMPS_position_map_radec_2.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_2.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_2.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_2.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_2.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_2.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_2.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_2.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_2.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_2.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_2.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_2.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_2.npy,13.05641491693826,-0.6189372458073498
data/example_generator_magrot/survey_PMPS_position_map_radec_3.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_3.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_3.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_3.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_3.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_3.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_3.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_3.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_3.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_3.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_3.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_3.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_3.npy,12.299076654328934,-1.4327208843495762
data/example_generator_magrot/survey_PMPS_position_map_radec_4.npy,data/example_generator_magrot/survey_SMPS_position_map_radec_4.npy,data/example_generator_magrot/survey_HTRU_position_map_radec_4.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vra_4.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vra_4.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vra_4.npy,data/example_generator_magrot/survey_PMPS_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_SMPS_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_HTRU_velocity_map_vdec_4.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_4.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_4.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_4.npy,data/example_generator_magrot/survey_PMPS_ppdot_map_fluxes_4.npy,data/example_generator_magrot/survey_SMPS_ppdot_map_fluxes_4.npy,data/example_generator_magrot/survey_HTRU_ppdot_map_fluxes_4.npy,13.286135700678276,-1.1544136490794008
...
```

In this way the format is easily compatible with our machine-learning pipeline.

The JSON file contains information about the mean, standard deviation, minimum and maximum values for each of the dataset labels.
This statistical information will be used during the training process if one wants to normalize or standardize the label values.

You can also choose to split the dataset into training/validation, training/test or into training/validation/test sets.
To do this you can run the `dataset_splitter.py` script in the `pypopsyn/generator` folder.
To generate a dataset split into two subsets, one specifically for training and the other for validation, you can specify a fraction of the total dataset that will form the validation subset by passing the argument `valid_split` in the `dataset_splitter` script. For example:
```commandline
python pypopsyn/generator/dataset_splitter.py --dataset_path generated_dataset --valid_split 0.2
```
This will create two files `dataset_train.csv` and `dataset_valid.csv` that will specify the samples belonging to the train dataset (80 % of the total dataset in this case) and the ones belonging to the validation dataset  (20 % of the total dataset).
The split is performed by randomly sampling the validation subset from the total dataset according to the specified split fraction.
In this case, the `statistics_train.json` file will contain the statistics computed on the labels of the training set only.

Analogously, if you want to split the dataset into two subsets, one specifically for training and the other for testing, you can specify a fraction of the total dataset that will form the test subset by passing the argument `test_split` in the `dataset_splitter` script. For example:
```commandline
python pypopsyn/generator/dataset_splitter.py --dataset_path generated_dataset --test_split 0.2
```
This will create two files `dataset_train.csv` and `dataset_test.csv` that will specify the samples belonging to the train dataset (80 % of the total dataset in this case) and the ones belonging to the test dataset  (20 % of the total dataset).
The split is performed by randomly sampling the test subset from the total dataset according to the specified split fraction.
In this case, the `statistics_train.json` file will contain the statistics computed on the labels of the training set only.

If you also want to create a test set in addition to the training and validation sets, you can specify the argument `test_split`, which sets the fraction of the total dataset to be dedicated for testing purposes.
In this case, the `valid_split` argument will specify the fraction of the dataset not used for testing but instead dedicated for validation.
```commandline
python pypopsyn/generator/dataset_splitter.py --dataset_path generated_dataset --test_split 0.1 --valid_split 0.2
```

This will create three files `dataset_train.csv`, `dataset_valid.csv` and `dataset_test.csv` that will specify the samples belonging to the train dataset (80 % of the dataset not used for testing in this case), the ones belonging to the validation dataset  (20 % of the dataset not used for testing) and the ones belonging to the test set (10 % of the total dataset), respectively.
Again the split is performed by randomly sampling the test and validation subsets from the dataset according to the specified split fractions.
In this case, the `statistics_train.json` file will also contain the statistics computed on the labels of the training set only.

!!! example

    To see a tutorial example for the generator you can look at the notebook in `tutorials/tutorial_notebooks/05_generator_tutorial.ipynb`.


# Generate ATNF maps

In order to perform inference on the observed data in the [ATNF catalog](https://www.atnf.csiro.au/research/pulsar/psrcat/) we need to produce the same maps used to train the network but for the observed sample.
To do this, one can use the following command:
```commandline
python pypopsyn/generator/generate_observed_data.py --path_atnf data/observations/atnf_full_nobinary_06-08-2024.csv --path_meerkat data/observations/meerkat_tpa_posselt_2023.csv --save_dir data/example_generator_observed --resolution_dyn 32 --resolution_ppdot 32
```
This will read the files `atnf_full_nobinary_06-08-2024.csv` and `meerkat_tpa_posselt_2023.csv` in the directory `data/observations` and generate the maps.
As for the scripts above, you can specify the type of the maps (either `array` or `image`) with the argument `--data_type` and the resolution with the arguments `--resolution_dyn` and `resolution_ppdot`.
Note that in order for the inference to work with a specific trained model the type and resolution of the maps generated from the ATNF catalog has to match the type and resolution of the simulated maps used to train the model neural network.

This script will generate 9 maps in total:
* Three position density maps in ICRS frame (one for each of the three radio surveys modeled by the simulator).
* Three $P-\dot{P}$ density maps (one for each of the three radio surveys modeled by the simulator).
* Three $P-\dot{P}$ density maps weighted with the logarithm of the radio flux (one for each of the three radio surveys modeled by the simulator).

* Moreover a `dataset_atnf.csv` will be created containing summary information of the dataset generated by this method.

!!! example

    To see a tutorial example for this generator you can look at the notebook in `tutorials/tutorial_notebooks/06_generator_observation_tutorial.ipynb`.
