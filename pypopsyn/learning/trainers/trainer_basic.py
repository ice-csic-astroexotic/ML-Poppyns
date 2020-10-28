"""
Basic trainer.

Authors:

    Alberto Garcia Garcia (garciagarcia@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import typing

import numpy as np
import torch
from torchvision.utils import make_grid

import pypopsyn.learning.utils as learning_utils
import pypopsyn.learning.utils.metric_tracker

from .trainer_base import BaseTrainer


class TrainerBasic(BaseTrainer):
    """
    Basic trainer.

    This class represents the most simple basic training pipeline which allows
    the user to perform training epochs coupled with validation passes and fully
    customize every single step of the pipeline (model to use, criterion to
    optimize, metrics to compute, optimizer to update the weights, and loaders
    from which data and targets can be fetched).

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: pypopsyn.learning.losses.loss_base,
        metric: pypopsyn.learning.metrics.metric_base,
        optimizer: torch.optim.Optimizer,
        configuration: pypopsyn.learning.configuration_parser,
        dataset_loader: pypopsyn.learning.loaders.loader_base,
        lr_scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        """
        Basic trainer initialization.

        Args:
            model (torch.nn.Module): network model to train.
            criterion (pypopsyn.LossBase): criterion for the loss calculation.
            metric (pypopsyn.MetricBase): accuracy metric to be computed.
            optimizer (torch.optim.Optimizer): optimizer for training.
            configuration (pypopsyn.learning.configuration_parser): config dict.
            dataset_loader (pypopsyn.learning.loaders.loader_base): dataset loader.
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler.

        Returns:
            Nothing.

        """

        super().__init__(model, criterion, metric, optimizer, configuration)

        # Define the loader for the total dataset.
        self.dataset_loader = dataset_loader

        self.logger.info(
            "Dataset loader normalization: {}".format(
                self.dataset_loader.normalize
            )
        )
        self.logger.info(
            "Dataset loader standardization: {}".format(
                self.dataset_loader.standardize
            )
        )

        if self.dataset_loader.normalize and self.dataset_loader.standardize:
            self.logger.error(
                "Standardization and normalization enabled for dataset loader..."
            )
            exit()

        # Fetch standardization and normalization factors for the targets.
        self.target_max = torch.tensor(self.dataset_loader.target_max).to(
            self.device
        )
        self.target_min = torch.tensor(self.dataset_loader.target_min).to(
            self.device
        )
        self.target_std = torch.tensor(self.dataset_loader.target_std).to(
            self.device
        )
        self.target_mean = torch.tensor(self.dataset_loader.target_mean).to(
            self.device
        )

        # Define the loaders for training and validation.
        self.train_loader = self.dataset_loader.train_loader
        self.valid_loader = self.dataset_loader.valid_loader

        # Define the total length of a training epoch
        self.len_epoch = len(self.train_loader)

        self.lr_scheduler = lr_scheduler

        self.log_step = int(np.sqrt(self.dataset_loader.batch_size))

        self.train_metrics = learning_utils.metric_tracker.MetricTracker(
            [], writer=self.writer
        )
        self.valid_metrics = learning_utils.metric_tracker.MetricTracker(
            [], writer=self.writer
        )

    def _train_epoch(self, epoch: int) -> typing.Tuple[dict, dict, dict]:
        """
        Single-epoch training routine.

        Args:
            epoch: Current epoch number.

        Returns:
            dict: a dictionary with the results for the epoch, i.e., the
            average for the losses and for the tracked metric for the training
            set.
            dict: the same but for the validation set (if available, None is
            returned otherwise).
            dict: a dictionary with the values for each individual loss for
            each one of the targets. If validation is performed, such losses
            correspond to validation losses, otherwise they are the training
            set losses.

        """

        # Set the model on training mode and reset all tracked metrics to zero.
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_loader):

            # Fetch data and labels and move them to the appropriate device.
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients to reset loss.
            self.optimizer.zero_grad()

            # Compute output for this batch.
            output = self.model(data)

            # De-normalize or de-standardize targets and outputs on the fly
            # if needed to rescale the loss values to a more readable range.
            # TODO: THIS COULD BE IMPROVED AND IDEALLY I WOULD LIKE THIS TO
            # BE DONE MORE TRANSPARENTLY, I DON'T KNOW HOW NOW.
            if self.dataset_loader.normalize:
                output_rescaled = (
                    output * (self.target_max - self.target_min)
                    + self.target_min
                )
                target_rescaled = (
                    target * (self.target_max - self.target_min)
                    + self.target_min
                )
            elif self.dataset_loader.standardize:
                output_rescaled = output * self.target_std + self.target_mean
                target_rescaled = target * self.target_std + self.target_mean

            # Compute each individual loss on each of the parameters to be
            # predicted by comparing the output and the ground truth for each
            # one of them. Then accumulate each individual loss in the total one.
            loss = 0.0
            loss_rescaled = 0.0
            for i in range(len(output[0])):
                # Compute individual loss for this output.
                loss_i = self.criterion(output[:, i], target[:, i])
                loss_rescaled_i = self.criterion(
                    output_rescaled[:, i], target_rescaled[:, i]
                )
                # Update tracked loss and output to TensorBoard.
                self.train_metrics.update(
                    "{}".format(self.dataset_loader.target_names[i]),
                    loss_rescaled_i.item(),
                )
                # Accumulate into total loss.
                loss = loss + loss_i
                loss_rescaled = loss_rescaled + loss_rescaled_i

            # Update tracked general loss and output to TensorBoard.
            self.train_metrics.update("loss", loss_rescaled.item())

            # Only backpropagate on total not-rescaled loss not on invidiual ones.
            loss.backward()
            self.optimizer.step()

            # Now output all the log information to console and write the
            # necessary log values for TensorBoard.

            # Set the TensorBoard step.
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Update tracked accuracy metric and output to TensorBoard.
            self.train_metrics.update(
                self.metric.__class__.__name__,
                self.metric(output_rescaled, target_rescaled),
            )

            # For each specified logging to console step, show the current
            # epoch training information (batch progress, loss...). Usually
            # We don't show it every batch because there will be too many.
            if batch_idx % self.log_step == 0:

                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch,
                        self._progress(
                            batch_idx, self.train_loader, self.len_epoch
                        ),
                        loss.item(),
                    )
                )

            if batch_idx == self.len_epoch:
                break

        # After a whole epoch has been carried out, store the dictionary of
        # results for each tracked metrics: usually the average loss and any
        # other specified accuracy metrics.
        log = self.train_metrics.result()
        # Pack the individual losses separatedly.
        losses = dict(
            filter(
                lambda e: e[0] in self.dataset_loader.target_names, log.items()
            )
        )

        # If there is a validation set, perform a validation step and fetch
        # the logged metrics and losses.
        val_log = self._valid_epoch(epoch)
        # Pack the individual losses separatedly.
        losses = dict(
            filter(
                lambda e: e[0] in self.dataset_loader.target_names,
                val_log.items(),
            )
        )

        # Step learning rate if a scheduler is provided.
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log, val_log, losses

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Single-epoch validation routine.

        Args:
            epoch (int): Current epoch number.

        Returns:
            A dictionary which contains the results of the validation over the
            whole dataset for all the requested metrics and losses.

        """

        # Set the model to evaluation mode and reset validation metrics.
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.valid_loader):

                # Fetch data and targets, move them to the compute device.
                data, target = data.to(self.device), target.to(self.device)

                # Compute predictions.
                output = self.model(data)

                # De-normalize or de-standardize targets and outputs on the fly
                # if needed to rescale the loss values to a more readable range.
                # TODO: THIS COULD BE IMPROVED AND IDEALLY I WOULD LIKE THIS TO
                # BE DONE MORE TRANSPARENTLY, I DON'T KNOW HOW NOW.
                if self.dataset_loader.normalize:
                    output = (
                        output * (self.target_max - self.target_min)
                        + self.target_min
                    )
                    target = (
                        target * (self.target_max - self.target_min)
                        + self.target_min
                    )
                elif self.dataset_loader.standardize:
                    output = output * self.target_std + self.target_mean
                    target = target * self.target_std + self.target_mean

                # Compute each individual loss on each of the parameters to be
                # predicted by comparing the output and the ground truth for
                # each one of them. Then accumulate each individual loss in the
                # total one which will be reported.
                loss = 0.0
                for i in range(len(output[0])):
                    # Compute individual loss for this output.
                    loss_i = self.criterion(output[:, i], target[:, i])
                    # Update tracked loss and output to TensorBoard.
                    self.valid_metrics.update(
                        "{}".format(self.dataset_loader.target_names[i]),
                        loss_i.item(),
                    )
                    # Accumulate into total loss.
                    loss = loss + loss_i

                # Update tracked loss and output to TensorBoard.
                self.valid_metrics.update("loss", loss.item())
                # Update tracked metric and output to TensorBoard.
                self.valid_metrics.update(
                    self.metric.__class__.__name__, self.metric(output, target)
                )

                # Set TensorBoard step.
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_loader) + batch_idx,
                    "validation",
                )

        # Add histogram of model parameters to TensorBoard.
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()
