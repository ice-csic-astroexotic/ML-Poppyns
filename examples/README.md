# MAGNESIA Population Synthesis Examples

This file is intended to gather information about all the examples available and
how to run them properly.

## Simulator

The `examples/simulator/initialize_evolve_population.py` script is responsible for initializing the simulated population from some initial conditions parameters and evolve it in time
To simulate populations with different initial parameters the package `hydra` is used. One and pass the values of the parameters through a configuration available at `pypopsyn/simulator/configuration.py` or directly on the command prompt. As an example to simulate four population with different kick velocities `vp_mean` values one can run the script:

```
python examples/simulator/initialize_evolve_population.py vp_mean=100.,200.,300.,600. -m 
```

In this way the multirun mode is used and a directory structure `multirun/date/time/simulated_population_index` is created, where `date` and `time` folders names are the date and time when the simulation has been performed and the `simulated_population_index` folders names are `0, 1, 2, ...`. The output files for the four simulated populations are stored respectively in the corresponding folders.

## Generator

### Generating Dataset

The `examples/generator/generating_dataset.py` script is responsible for, given a set of final evolved populations executed via a multirun, generating a heatmap/density map for each one of them and creating a CSV file to represent the whole dataset and make it easy for later loading it into the learning subpackage. Such CSV provides one line for each sample in the dataset in which we indicate the file path for its density map and the values for its parameters like this:

```
filename,r_extent
examples/data/2020-03-23/11-42-24/train_set/density_map_pop_0.png,30.0
examples/data/2020-03-23/11-42-24/train_set/density_map_pop_1.png,40.0
examples/data/2020-03-23/11-42-24/train_set/density_map_pop_2.png,50.0
...
```

In order to generate a dataset out of a set of simulated and evolved populations you will first need to run the `examples/simulator/initialize_evolve_population.py` example (see above for more information about it). After that, you can call this script providing the date and time when the simulated samples were created and the name of the dataset where to save the density maps:

```
python examples/generator/generating_dataset.py --date="2020-03-23" --time="11-42-24" --dataset_name="dataset1"
```

That will create a folder `examples/data/2020-03-23/11-42-24/dataset1` in which a `dataset.csv` file with all the information will be created and each individual `density_map_popX.png` will be saved.

## Learning

The `examples/learning/train.py` script allows to train a neural network over a sample of simulated neutron stars populations.
Once the datset containing the density maps has been created, to train the network over the dataset one can run the script:

```
python examples/learning/train.py --configuration="config.json"
```
where the config.json file contains all the information needed by the network to train. In particular in this file one can specify the network model architecture to use:
```
"arch": {
    "type": "ModelNN",
    "args": {}
  },
```
the loader for the dataset, the path to the folder containing the dataset, the batch size ecc.:
```
"data_loader": {
   "type": "Loader",
   "args": {
     "data_path": "examples/data/2020-03-23/11-42-24/train_set/dataset.csv",
     "batch_size": 4,
     "num_workers": 1
   }
 },
```
the loss function to use and the metric to monitor the accuracy of the neural network model: 
```
"loss": {
      "type": "Loss",
      "args": {}
  },

  "metric": {
      "type": "MetricAccuracy",
      "args": {}
  },
```
The implemeted loss function `LossRMSE` evaluates the average root mean square error (RMSE) between the output of the network and the target labels over every batch.
For the accuracy metric a reduced chi square metric is implemented called `MetriAccuracyCHI2`. An optimal value of the reduced chi square should be around 1 for a well trained network.
