# Population Synthesis Learning

This is the learning subpackage for the population synthesis code. It contains
all the needed tools to construct machine learning pipelines with PyTorch to
deal with data coming form the simulator subpackage which in turn has been
previously processed by the generator subpackage.

In particular, this learning package contains (or provides the means to
implement):

* Custom layers to be used in the network.
* Data loaders to fetch any kind of data.
* A logging subsystem to produce messages to file and console.
* Losses to use during the training process.
* Metrics to evaluate the results with.
* Models that make use of custom or standard layers.
* Trainers that drive the training process.
* Various utils that help during the process.
* A configuration parser class for customizing the experiments.

An example that makes use of this subpackage to create a network and train it
by feeding data through a data loader is located in `examples/learning/train.py`.

## Loaders

Loaders provide data to the training pipeline while supporting batching,
transforms and other complex operations.

They live in the `/loaders` folder and its corresponding `loaders` module. All
loaders derive from the abstract base class `LoaderBase` in `loader_base.py`
which in turn derives from `torch.utils.data.DataLoader`; this ensures that
it can be seamlessly used in any PyTorch pipeline for data loading.

To create a new data loader, just create a new `loader_xxxx.py` file inside the
`/loaders` folder, inside it create a new class which inherits from `LoaderBase`
and register the loader to be used in the `loaders.py`.

As an example, we provide the `loader_mnist.py` to fetch and load the MNIST
dataset for image classification.

## Models

Models live in the `/models` folder and its corresponding `models` module. All
of them should derive from the abstract base class `ModelBase` in `model_base.py`
which lays out the basic interface any model must implement.

In order to create a new model, just create a new `model_xxxx.py` file inside
the `/models` folder, create a new class which inherits from `ModelBase` and
register your model class to be used in the `models.py` module by importing it.

If you need an example, check how `model_mnist.py` is implemented and registered
in the `models.py` file.

## Losses

Losses live in the `/losses` folder and its corresponding `losses` module thus
following the same structure as the modules. All losses must derive from the
base class `LossBase` in `loss_base.py`, the basic interface. Note that losses
themselves are callable classes.

To create a new loss, just create a new `loss_xxxx.py` file inside the `/losses`
folder, create a new class which inherits from `LossBase` and register your
loss class to be used in the `losses.py` module by importing it.

An example Negative Log-Likelihood is already implemented in `loss_nll.py` and
also registered in the `losses.py` file.

## Metrics

Metrics live in the `/metrics` folder and its corresponding `metrics` module thus
following the same structure as the modules and losses. All metrics must derive
from the base class `MetricBase` in `metric_base.py`, the basic interface.
Note that metrics themselves are callable classes.

To create a new metric, just create a new `metric_xxxx.py` file inside the 
`/metrics` folder, create a new class which inherits from `MetricBase` and
register your metric class to be used in the `metrics.py` module by importing it.

An example Accuracy metric is already implemented in `metric_accuracy.py` and
also registered in the `metrics.py` file.

## Trainers

TODO

## Disclaimer

Huge thanks to Victor Huang and Seonkyu Park for their contributions in the 
awesome [PyTorch template](https://github.com/victoresque/pytorch-template).