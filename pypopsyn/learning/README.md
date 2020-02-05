## Models

Models live in the `/models` folder and its corresponding `models` module. All
of them should derive from the abstract base class `ModelBase` in `model_base.py`
which lays out the basic interface any model must implement.

In order to create a new model, just create a new `model_xxxx.py` file inside
the `/models` folder, create a new class which inherits from `ModelBase` and
register your model class to be used in the `models.py` module by importing it.

If you need an example, check how `model_mnist.py` is implemented and registered
in the `models.py` file.

## Disclaimer

Huge thanks to Victor Huang and Seonkyu Park for their contributions in the 
awesome [PyTorch template](https://github.com/victoresque/pytorch-template).