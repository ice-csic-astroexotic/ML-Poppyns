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
examples/data/train_set/density_map_pop_0.png,30.0
examples/data/train_set/density_map_pop_1.png,40.0
examples/data/train_set/density_map_pop_2.png,50.0
...
```

In order to generate a dataset out of a set of simulated and evolved populations you will first need to run the `examples/simulator/generate_evolve_population.py` example (see above for more information about it). After that, you can call this script providing the path to the multirun folder where the final populations were generated and the generic filename for each one of them:

```
python examples/generator/generating_dataset.py --dataset_name dataset1 --root_path=multirun/2020-03-18/11-11-19/ --file_name=final_population.txt
```

That will create a folder `examples/data/dataset1` in which a `dataset.csv` file with all the information will be created and each individual `density_map_popX.png` will be saved.

## Learning

TODO