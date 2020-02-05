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

## Disclaimer

Huge thanks to Victor Huang and Seonkyu Park for their contributions in the 
awesome [PyTorch template](https://github.com/victoresque/pytorch-template).