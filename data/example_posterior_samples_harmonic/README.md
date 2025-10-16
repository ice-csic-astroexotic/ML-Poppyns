# Unnormalized posterior samples to be used with the Harmonic package for model comparison

This is an example of samples from the unnormalized posterior distribution obtained by running an SBI inference using 
SNLE as the model type. These samples can then be used for model comparison with the Harmonic package, as shown
in the notebook `tutorials/analysis_notebooks/model_comparison_sbi.ipynb`.

To run this example, we use the following command:
```commandline
python mlpoppyns/learning/sbi_train.py
```
By default, this script uses the configuration specified in `mlpoppyns/learning/config_sbi.json`. To use `SNLE`, you need 
to change the `type` to "snle" in the trainer section of this file.