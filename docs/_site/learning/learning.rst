********
Learning
********

Weight initializers
####################

Initializer parent class
************************
.. automodule:: pypopsyn.learning.initializers.initializer_base
  :members: initializer_base

Kaiming initializer
*******************
.. automodule:: pypopsyn.learning.initializers.initializer_kaiming
  :members: initializer_kaiming

Normal initializer
******************
.. automodule:: pypopsyn.learning.initializers.initializer_normal
  :members: initializer_normal

Uniform initializer
*******************
.. automodule:: pypopsyn.learning.initializers.initializer_uniform
  :members: initializer_uniform

Uniform rule initializer
************************
.. automodule:: pypopsyn.learning.initializers.initializer_uniform_rule
  :members: initializer_uniform_rule

Xavier initializer
******************
.. automodule:: pypopsyn.learning.initializers.initializer_xavier
  :members: initializer_xavier

Layers
######

Dataset loaders
###############

Loader parent class
*******************
.. automodule:: pypopsyn.learning.loaders.loader_base
  :members: loader_base

Multichannel array loader
*************************
.. automodule:: pypopsyn.learning.loaders.loader_multichannel_array_stat
  :members: loader_multichannel_array

Multichannel image loader
*************************
.. automodule:: pypopsyn.learning.loaders.loader_multichannel_image
  :members: loader_multichannel_image

RGB image loader
****************
.. automodule:: pypopsyn.learning.loaders.loader_rgb_image
  :members: loader_rgb_image

Logger and Tensorboard
######################

Logger
******
.. automodule:: pypopsyn.learning.logger.logger
  :members: logger

Tensorboard
***********
.. automodule:: pypopsyn.learning.logger.tensorboard_writer
  :members: tensorboard_writer

Losses
######

Metric monitoring and update
****************************
.. automodule:: pypopsyn.learning.metrics.metric_accuracy
  :members: metric_accuracy

Loss parent class
*****************
.. automodule:: pypopsyn.learning.losses.loss_base
  :members: loss_base

Mean absolute error (MAE) loss
**********************************
.. automodule:: pypopsyn.learning.losses.loss_mae
  :members: loss_mae

Mean square error (MSE) loss
**********************************
.. automodule:: pypopsyn.learning.losses.loss_mse
  :members: loss_rmse

Negative log-likelihood loss
****************************
.. automodule:: pypopsyn.learning.losses.loss_nll
  :members: loss_nll

Root mean square error (RMSE) loss
**********************************
.. automodule:: pypopsyn.learning.losses.loss_rmse
  :members: loss_rmse

Accuracy metrics
################

Metric parent class
*******************
.. automodule:: pypopsyn.learning.metrics.metric_base
  :members: metric_base

Chi square error metric
***********************
.. automodule:: pypopsyn.learning.metrics.metric_chi2
  :members: metric_chi2

Mean absolute error (MAE) metric
********************************
.. automodule:: pypopsyn.learning.metrics.metric_mae
  :members: metric_mae

Mean square error (MSE) metric
******************************
.. automodule:: pypopsyn.learning.metrics.metric_mse
  :members: metric_mse

Root mean square error (RMSE) metric
************************************
.. automodule:: pypopsyn.learning.metrics.metric_rmse
  :members: metric_rmse

Neural network architecture models
##################################

Model parent class
******************
.. automodule:: pypopsyn.learning.models.model_base
  :members: model_base

Linear fully connected model
****************************
.. automodule:: pypopsyn.learning.models.model_linear
  :members: model_linear

Convolutional model
*******************
.. automodule:: pypopsyn.learning.models.model_conv
  :members: model_conv

Convolutional model for sbi
***************************
.. automodule:: pypopsyn.learning.models.model_conv_sbi
  :members: model_conv_sbi

Deeper Convolutional model for sbi
**********************************
.. automodule:: pypopsyn.learning.models.model_conv_sbi_deep
  :members: model_conv_sbi_deep

Trainers
########

Trainer parent class
********************
.. automodule:: pypopsyn.learning.trainers.trainer_base
  :members: trainer_base

Basic trainer
*************
.. automodule:: pypopsyn.learning.trainers.trainer_basic
  :members: trainer_basic

Utils
#####

Benchmark model performance
***************************
.. automodule:: pypopsyn.learning.utils.benchmark
  :members: benchmark

JSON file reader
****************
.. automodule:: pypopsyn.learning.utils.json
  :members: json

Metric tracker
****************
.. automodule:: pypopsyn.learning.utils.metric_tracker
  :members: metric_tracker

Request GPU devices
*******************
.. automodule:: pypopsyn.learning.utils.request_device
  :members: request_device

Update configuration from CLI
#############################

Configuration parser
********************
.. automodule:: pypopsyn.learning.configuration_parser
  :members: configuration_parser

Train and inference scripts
###########################

train script
************
.. automodule:: pypopsyn.learning.train
  :members: train

train script for sbi (amortized)
********************************
.. automodule:: pypopsyn.learning.train_sbi
  :members: train_sbi

train script for sbi (truncated SNPE)
*************************************
.. automodule:: pypopsyn.learning.train_tsnpe
  :members: train_tsnpe

Infer script
************
.. automodule:: pypopsyn.learning.infer
  :members: infer

Infer script for sbi
********************
.. automodule:: pypopsyn.learning.infer_sbi
  :members: infer_sbi

Infer script for sbi (ensemble method)
**************************************
.. automodule:: pypopsyn.learning.infer_sbi_ensemble
  :members: infer_sbi_ensemble
