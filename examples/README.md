# MAGNESIA Population Synthesis Examples

This file is intended to gather information about all the examples available and
how to run them properly.

## Simulator

TODO

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

TODO
