# Inference example (sbi)
This is an example of inference using the simulation-base inference framework.

To run this example we use the following command:
```commandline
python pypopsyn/learning/infer_sbi.py --trained_model data/example_learning_sbi/models/SBI_ConvolutionMDN/20240626_105721/trained_model.pickle --corner_plot True
```
By default this script will use the configuration specified in `pypopsyn/learning/config_sbi.json`.