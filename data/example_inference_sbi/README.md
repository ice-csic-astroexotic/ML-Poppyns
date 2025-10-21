# Inference example (sbi)

This is an example of statistical inference using the simulation-based inference framework from the sbi library.

To run this example, we use the following command:
```commandline
python mlpoppyns/learning/sbi_infer.py --trained_model data/example_learning_sbi/models/SBI_ConvolutionMDN/20240626_105721/trained_model.pickle --corner_plot True
```
By default, this script will use the configuration specified in `mlpoppyns/learning/config_sbi.json`.