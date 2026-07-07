# Generating density maps

## Maps of synthetic populations

Once a set of synthetic populations has been created by running the `run_simulation_set.py` script with one of the 
simulator types described in the section [Simulating neutron star populations](simulator_tutorial.md), we can generate a synthetic 
representation of these simulations that is readable by a machine-learning pipeline. Depending on the type of 
simulation that has been performed, three types of generator scripts

* `mlpoppyns/generator/generate_dataset_full.py`
* `mlpoppyns/generator/generate_dataset_surveys.py`

can be used to produce two-dimensional density maps of our population(s).

We use the first script `mlpoppyns/generator/generate_dataset_full.py` for end-to-end simulations that were obtained
using the `simulate_population_full.py` module. The script will read the corresponding `final_population.pkl.gz` 
output files for each simulated population and generate density maps in the form of either `.png` images or 2D 
NumPy arrays. To run this generator script, we have to specify the following parameters:

* `data`: the path where the simulated populations are located.
* `save_dir`: the path to the folder where the dataset of maps is saved.
* `data_type`: the type of maps to produce, i.e., either `array` or `image`. 
    If `array`, the generator will produce the density maps in the form of 2D `.npy` arrays; 
    if `image`, density maps are produced in the form of `.png` images;
    if `array_kde` or `image_kde`, the generator will perform a kernel density estimation (KDE) before generating the maps.
* `resolution_dyn`: the resolution in bins or pixels of the maps containing the dynamical information.
* `resolution_ppdot`: the resolution in bins or pixels of the maps containing the $P-\dot{P}$ information.

The output produced by this script contains the spatial density and velocity information for all evolved neutron stars 
as well as their distribution in the $P-\dot{P}$ plane. More precisely, we produce the following density maps:

* 1 map for stellar positions in galactocentric $x$ and $y$ coordinates.
* 1 map for stellar positions in galactocentric $x$ and $z$ coordinates.
* 1 map in galactocentric $x$ and $y$ coordinates weighted by the radial velocity component $v_r$.
* 1 map in galactocentric $x$ and $y$ coordinates weighted by the azimuthal velocity $v_{\phi}$.
* 1 map in galactocentric $x$ and $y$ coordinates weighted by the velocity component $v_z$. 
* 1 map for stellar positions in ICRS coordinates.
* 1 map in ICRS coordinates weighted by the proper motion component $\mu_{\rm RA}$.
* 1 map in ICRS coordinates weighted by the proper motion component $\mu_{\rm DEC}$.
* 1 $P-\dot{P}$ map.

The second script `mlpoppyns/generator/generate_dataset_surveys.py` is used for simulations that have been run using 
the `simulate_population_magrot_det.py` module. The script will read the corresponding `.pkl.gz` output files that are
produced for each of our simulated surveys for each synthetic population. It then generates a set of density maps in 
the form of either `.png` images or 2D NumPy arrays for those simulated neutron stars that are detected by the surveys
only. The corresponding density maps store their spatial density and proper motion information in the equatorial 
(ICRS) reference frame and their distribution and corresponding fluxes in the $P-\dot{P}$ plane. More precisely, we 
produce the following density maps (one for each of our three radio surveys modeled by the simulator) for each of the
simulated populations:

* 4 maps for stellar positions in ICRS coordinates.
* 4 maps in ICRS coordinates weighted by the proper motion component $\mu_{\rm RA}$.
* 4 maps in ICRS coordinates weighted by the proper motion component $\mu_{\rm DEC}$.
* 4 $P-\dot{P}$ maps.
* 4 $P-\dot{P}$ maps, 3 weighted by the logarithm of the radio flux and 1 weighted by the logarithm of the X-ray flux.

When running this generator script, we have to specify the following parameters:

* `data`: the path where the simulated populations are located.
* `save_dir`: the path to the folder where the dataset of maps is saved.
* `data_type`: the type of maps to produce either `array`, `array_kde`, `image` or `image_kde`. 
    If `array`, the generator will produce the density maps in the form of 2D `.npy` arrays; 
    if `image`, density maps are produced in the form of `.png` images;
    if `array_kde` or `image_kde`, the generator will perform a kernel density estimation (KDE) before generating the maps.
* `resolution_dyn_radio`: the resolution in bins or pixels of the maps containing the dynamical information for the 
   neutron stars detected in radio.
* `resolution_ppdot_radio`: the resolution in bins or pixels of the maps containing the $P-\dot{P}$ information for 
   the neutron stars detected in radio.
* `resolution_dyn_xray`: the resolution in bins or pixels of the maps containing the dynamical information for the 
   neutron stars detected in X-ray.
* `resolution_ppdot_xray`: the resolution in bins or pixels of the maps containing the $P-\dot{P}$ information for the 
   neutron stars detected in X-ray.
* `generate_xray`: option specifying if the maps related to the X-ray surveys are produced.
* `filter_young_xdins`: option specifying if only young magnetars and XDINSs are considered in the X-ray population.

Suppose that we have created a dataset of simulated populations that is stored in `data/example_simulation_helper_magrot` 
using the `simulate_population_magrot_det.py` script. Let us assume that we want to create a map dataset of 2D arrays 
for the spatial and velocity information with a resolution of $32 \times 16$ and the density and radio flux maps in 
the $P-\dot{P}$ plane with a resolution of $32 \times 32$. To obtain these maps, we run the following command:
```commandline
python mlpoppyns/generator/generate_dataset_surveys.py --data data/example_simulation_helper_magrot --save_dir output/generator --data_type array --resolution_dyn_radio 32 --resolution_ppdot_radio 32 --resolution_dyn_xray 32 --resolution_ppdot_xray 32 --generate_xray
```
By running the script, we create a folder `output/generator` where a set of 2D arrays is stored for each simulated 
population (sample). In addition, this produces a single `dataset_full.csv` and `statistics_full.json` file containing 
information about the dataset and statistical information for each label, respectively. 

The CSV file provides one line for each synthetic simulation sample in the dataset indicating the file path for its 
2D arrays (these are potential input channels for the machine learning pipeline) and the numerical values used to 
generate the underlying population (these will serve as labels for the machine learning). 

A `dataset_full.csv` file for a set of five synthetic simulations could, for example, look like this:
```commandline
input:survey_PMPS_density_map_radec,input:survey_SMPS_density_map_radec,input:survey_HTRU_density_map_radec,input:survey_xray_density_map_radec,input:survey_PMPS_velocity_vra_map_radec,input:survey_SMPS_velocity_vra_map_radec,input:survey_HTRU_velocity_vra_map_radec,input:survey_xray_velocity_vra_map_radec,input:survey_PMPS_velocity_vdec_map_radec,input:survey_SMPS_velocity_vdec_map_radec,input:survey_HTRU_velocity_vdec_map_radec,input:survey_xray_velocity_vdec_map_radec,input:survey_PMPS_density_map_ppdot,input:survey_SMPS_density_map_ppdot,input:survey_HTRU_density_map_ppdot,input:survey_xray_density_map_ppdot,input:survey_PMPS_flux_map_ppdot,input:survey_SMPS_flux_map_ppdot,input:survey_HTRU_flux_map_ppdot,input:survey_xray_flux_map_ppdot,B_initial_log10_mean,P_initial_log10_mean
data/example_generator_magrot/survey_PMPS_density_map_radec_0.npy,data/example_generator_magrot/survey_SMPS_density_map_radec_0.npy,data/example_generator_magrot/survey_HTRU_density_map_radec_0.npy,data/example_generator_magrot/survey_xray_density_map_radec_0.npy,data/example_generator_magrot/survey_PMPS_velocity_vra_map_radec_0.npy,data/example_generator_magrot/survey_SMPS_velocity_vra_map_radec_0.npy,data/example_generator_magrot/survey_HTRU_velocity_vra_map_radec_0.npy,data/example_generator_magrot/survey_xray_velocity_vra_map_radec_0.npy,data/example_generator_magrot/survey_PMPS_velocity_vdec_map_radec_0.npy,data/example_generator_magrot/survey_SMPS_velocity_vdec_map_radec_0.npy,data/example_generator_magrot/survey_HTRU_velocity_vdec_map_radec_0.npy,data/example_generator_magrot/survey_xray_velocity_vdec_map_radec_0.npy,data/example_generator_magrot/survey_PMPS_density_map_ppdot_0.npy,data/example_generator_magrot/survey_SMPS_density_map_ppdot_0.npy,data/example_generator_magrot/survey_HTRU_density_map_ppdot_0.npy,data/example_generator_magrot/survey_xray_density_map_ppdot_0.npy,data/example_generator_magrot/survey_PMPS_flux_map_ppdot_0.npy,data/example_generator_magrot/survey_SMPS_flux_map_ppdot_0.npy,data/example_generator_magrot/survey_HTRU_flux_map_ppdot_0.npy,data/example_generator_magrot/survey_xray_flux_map_ppdot_0.npy,12.760581505895203,-0.7428330891684274
data/example_generator_magrot/survey_PMPS_density_map_radec_1.npy,data/example_generator_magrot/survey_SMPS_density_map_radec_1.npy,data/example_generator_magrot/survey_HTRU_density_map_radec_1.npy,data/example_generator_magrot/survey_xray_density_map_radec_1.npy,data/example_generator_magrot/survey_PMPS_velocity_vra_map_radec_1.npy,data/example_generator_magrot/survey_SMPS_velocity_vra_map_radec_1.npy,data/example_generator_magrot/survey_HTRU_velocity_vra_map_radec_1.npy,data/example_generator_magrot/survey_xray_velocity_vra_map_radec_1.npy,data/example_generator_magrot/survey_PMPS_velocity_vdec_map_radec_1.npy,data/example_generator_magrot/survey_SMPS_velocity_vdec_map_radec_1.npy,data/example_generator_magrot/survey_HTRU_velocity_vdec_map_radec_1.npy,data/example_generator_magrot/survey_xray_velocity_vdec_map_radec_1.npy,data/example_generator_magrot/survey_PMPS_density_map_ppdot_1.npy,data/example_generator_magrot/survey_SMPS_density_map_ppdot_1.npy,data/example_generator_magrot/survey_HTRU_density_map_ppdot_1.npy,data/example_generator_magrot/survey_xray_density_map_ppdot_1.npy,data/example_generator_magrot/survey_PMPS_flux_map_ppdot_1.npy,data/example_generator_magrot/survey_SMPS_flux_map_ppdot_1.npy,data/example_generator_magrot/survey_HTRU_flux_map_ppdot_1.npy,data/example_generator_magrot/survey_xray_flux_map_ppdot_1.npy,12.00901699270494,-0.7370723718542642
data/example_generator_magrot/survey_PMPS_density_map_radec_2.npy,data/example_generator_magrot/survey_SMPS_density_map_radec_2.npy,data/example_generator_magrot/survey_HTRU_density_map_radec_2.npy,data/example_generator_magrot/survey_xray_density_map_radec_2.npy,data/example_generator_magrot/survey_PMPS_velocity_vra_map_radec_2.npy,data/example_generator_magrot/survey_SMPS_velocity_vra_map_radec_2.npy,data/example_generator_magrot/survey_HTRU_velocity_vra_map_radec_2.npy,data/example_generator_magrot/survey_xray_velocity_vra_map_radec_2.npy,data/example_generator_magrot/survey_PMPS_velocity_vdec_map_radec_2.npy,data/example_generator_magrot/survey_SMPS_velocity_vdec_map_radec_2.npy,data/example_generator_magrot/survey_HTRU_velocity_vdec_map_radec_2.npy,data/example_generator_magrot/survey_xray_velocity_vdec_map_radec_2.npy,data/example_generator_magrot/survey_PMPS_density_map_ppdot_2.npy,data/example_generator_magrot/survey_SMPS_density_map_ppdot_2.npy,data/example_generator_magrot/survey_HTRU_density_map_ppdot_2.npy,data/example_generator_magrot/survey_xray_density_map_ppdot_2.npy,data/example_generator_magrot/survey_PMPS_flux_map_ppdot_2.npy,data/example_generator_magrot/survey_SMPS_flux_map_ppdot_2.npy,data/example_generator_magrot/survey_HTRU_flux_map_ppdot_2.npy,data/example_generator_magrot/survey_xray_flux_map_ppdot_2.npy,13.666001670349527,-0.945232045269706
data/example_generator_magrot/survey_PMPS_density_map_radec_3.npy,data/example_generator_magrot/survey_SMPS_density_map_radec_3.npy,data/example_generator_magrot/survey_HTRU_density_map_radec_3.npy,data/example_generator_magrot/survey_xray_density_map_radec_3.npy,data/example_generator_magrot/survey_PMPS_velocity_vra_map_radec_3.npy,data/example_generator_magrot/survey_SMPS_velocity_vra_map_radec_3.npy,data/example_generator_magrot/survey_HTRU_velocity_vra_map_radec_3.npy,data/example_generator_magrot/survey_xray_velocity_vra_map_radec_3.npy,data/example_generator_magrot/survey_PMPS_velocity_vdec_map_radec_3.npy,data/example_generator_magrot/survey_SMPS_velocity_vdec_map_radec_3.npy,data/example_generator_magrot/survey_HTRU_velocity_vdec_map_radec_3.npy,data/example_generator_magrot/survey_xray_velocity_vdec_map_radec_3.npy,data/example_generator_magrot/survey_PMPS_density_map_ppdot_3.npy,data/example_generator_magrot/survey_SMPS_density_map_ppdot_3.npy,data/example_generator_magrot/survey_HTRU_density_map_ppdot_3.npy,data/example_generator_magrot/survey_xray_density_map_ppdot_3.npy,data/example_generator_magrot/survey_PMPS_flux_map_ppdot_3.npy,data/example_generator_magrot/survey_SMPS_flux_map_ppdot_3.npy,data/example_generator_magrot/survey_HTRU_flux_map_ppdot_3.npy,data/example_generator_magrot/survey_xray_flux_map_ppdot_3.npy,12.597396839756216,-1.4013555393326107
...
```

!!! note

    The format of the `dataset_full.csv` file ensures direct compatibility with our machine learning pipeline.

The JSON file contains information about the mean, standard deviation, minimum and maximum values for each of the 
dataset labels computed across the entire dataset of simulations, and could look like the following:
```commandline
{
    "B_initial_log10_mean": {
        "max": 13.968959556720801,
        "mean": 12.766555558229868,
        "min": 12.187267435646975,
        "std": 0.5176380894348778
    },
    "P_initial_log10_mean": {
        "max": -0.331459108143384,
        "mean": -0.8783193891550154,
        "min": -1.4647713639094808,
        "std": 0.3379764548527923
    }
}
```

We use this statistical information during the training process if one wants to normalize or 
standardize the label values.

!!! example

    An example of this generator is presented in detail in the tutorial `tutorials/tutorial_notebooks/05_generator_tutorial.ipynb`.

## Training, validation and test splits

We can also split the dataset into training/validation, training/test or into training/validation/test sets. To do this,
we use the `dataset_splitter.py` script in the `mlpoppyns/generator` folder.

To split a dataset into two subsets, one for training and the other for validation, we specify a fraction of the total 
dataset that will form the validation subset by passing the argument `valid_split` in the `dataset_splitter.py` script. 
For example:
```commandline
python mlpoppyns/generator/dataset_splitter.py --dataset_path output/generator --valid_split 0.2
```
This will create two files `dataset_train.csv` and `dataset_valid.csv` in the location of the `dataset_full.csv` file
that will specify the specific simulation samples belonging to the train dataset (80% of the total dataset in this case)
and those belonging to the validation dataset (20% of the total dataset). This split is performed by randomly sampling 
the validation subset from the total dataset according to the specified split. We also produce a file
`statistics_train.json` (saved in the same location) that contains the statistics computed on the training labels only.

Analogously, if we want to split the dataset into two subsets, one for training and the other for testing, we specify 
a fraction of the total dataset that will form the test subset by passing the argument `test_split` in the 
`dataset_splitter.py` script. For example:
```commandline
python mlpoppyns/generator/dataset_splitter.py --dataset_path output/generator --test_split 0.2
```
The situation is identical to above apart from the fact that we now produce a `dataset_test.csv` file that contains 
information on 20% randomly sampled test simulations.

Finally, to create a test set in addition to the training and validation sets, we specify both split 
arguments. In this case, the argument `test_split` sets the fraction of the total dataset to be dedicated for testing 
purposes. The `valid_split` argument will then separate the remaining fraction of the dataset into validation and
training sets. For example
```commandline
python mlpoppyns/generator/dataset_splitter.py --dataset_path output/generator --test_split 0.1 --valid_split 0.2
```
generates a 10% test dataset. The remaining 90% will be split into 20% validation and 80% training, which equates to
a validation dataset size of 18% and training dataset size of 72% of the entire initial dataset. The corresponding 
simulation samples are saved in the `dataset_test.csv`, `dataset_valid.csv` and `dataset_train.csv`, respectively.
As above, the splits are performed by randomly sampling the test and validation subsets, while the 
`statistics_train.json` file contains the statistics computed on the labels of the training set only.

!!! example

    An example of this dataset splitter is also presented in detail in the tutorial `tutorials/tutorial_notebooks/05_generator_tutorial.ipynb`.


## Maps of the observed population

To perform inference on the observed data in the [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/)
combined with the radio fluxes from the TPA program on MeerKat [(Posselt et al., 2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4582P/abstract) and the catalogue of 
thermally-emitting X-ray neutron stars (Dehman et al. in prep.) with an optimized neural network, we need to convert 
the corresponding data into the same representation that was used to optimize our machine learning pipeline. That is, 
we need to produce the same kind of density maps as outlined above for the observed pulsar population.

To do this, we use the `mlpoppyns/generator/generate_observed_data.py` script. We can, for example, run:
```commandline
python mlpoppyns/generator/generate_observed_data.py --path_atnf data/observations/atnf_full_nobinary_24-09-2024_with_errors.csv --path_meerkat data/observations/meerkat_tpa_posselt_2023.csv --path_xray data/observations/thermal_NS_05-11-2024.csv --save_dir output/generator_observed --resolution_dyn_radio 32 --resolution_ppdot_radio 32 --resolution_dyn_xray 32 --resolution_ppdot_xray 32
```
This will read the files `atnf_full_nobinary_24-09-2024_with_errors.csv`, `meerkat_tpa_posselt_2023.csv` and `thermal_NS_05-11-2024.csv` in the directory 
`data/observations` and generate the maps.

As for the scripts above, we can specify the map type (either `array`, `array_kde`, `image` or `image_kde`) with the argument `--data_type` and 
the resolution with the arguments `--resolution_dyn_radio`, `--resolution_ppdot_radio`, `--resolution_dyn_xray` and `--resolution_ppdot_xray`. 

!!! note
    
    To sucessfully apply a trained model to our observed population, the type and resolution of the maps generated 
    from the observational data have to match the type and resolution of the simulated maps used to train the neural
    network.

In the above example, we then generate the following twelve maps:

* Four position density maps in ICRS coordinates: one for each of the three radio surveys and one for the X-ray survey 
   modeled by the simulator.
* Four $P-\dot{P}$ density maps: one for each of the three simulated radio surveys and one for the X-ray survey.
* Four $P-\dot{P}$ density maps weighted by the logarithm of the radio and X-ray fluxes: three for the three simulated 
   radio surveys and one for the X-ray survey.

Moreover, the above command also produces a `dataset_atnf.csv` file containing summary information of the different
maps. Note that the underlying ground truths are, of course, unknown here. The corresponding entries in the CSV file,
required for compatibility purposes, are therefore left empty.

!!! example

    To see a tutorial example for this generator you can look at the notebook in `tutorials/tutorial_notebooks/06_generator_observation_tutorial.ipynb`.
