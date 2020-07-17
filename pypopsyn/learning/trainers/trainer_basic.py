""" Basic trainer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import numpy as np
import torch
from torchvision.utils import make_grid

import pypopsyn.learning.utils as learning_utils
import pypopsyn.learning.utils.metric_tracker

from .trainer_base import BaseTrainer


class TrainerBasic(BaseTrainer):
    """
    Basic trainer.

    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        configuration,
        data_loader,
        validation_data_loader=None,
        lr_scheduler=None,
    ) -> None:
        """
        Basic trainer initialization.

        Args:
            model
            criterion
            metrics
            optimizer
            configuration
            data_loader
            validation_data_loader
            lr_scheduler

        Returns:
            Nothing.

        """

        super().__init__(model, criterion, metrics, optimizer, configuration)

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.validation_data_loader = validation_data_loader
        self.validate = self.validation_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = learning_utils.metric_tracker.MetricTracker(
            [], writer=self.writer
        )
        self.valid_metrics = learning_utils.metric_tracker.MetricTracker(
            [], writer=self.writer
        )

    def _train_epoch(self, epoch: int) -> dict:
        """
        Single-epoch training routine.

        Args:
            epoch: Current epoch number.

        Returns:
            A dictionary containing the results for the epoch, i.e., the average
            for each tracked metric: usually the loss average for the epoch, and
            any other specified accuracy metric average.

        """

        # Set the model on training mode and reset all tracked metrics to zero.
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Fetch data and labels and move them to the appropriate device.
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients to reset loss.
            self.optimizer.zero_grad()

            # Compute output for this batch.
            output = self.model(data)

            # Compute each individual loss on each of the parameters to be
            # predicted by comparing the output and the ground truth for each
            # one of them. Then accumulate each individual loss in the total one.
            loss = 0.0
            for i in range(len(output[0])):
                # Compute individual loss for this output.
                loss_i = self.criterion(output[:, i], target[:, i])
                # Update tracked loss and output to TensorBoard.
                self.train_metrics.update("loss{}".format(i), loss_i.item())
                # Accumulate into total loss.
                loss = loss + loss_i

            # Update tracked general loss and output to TensorBoard.
            self.train_metrics.update("loss", loss.item())

            # Only backpropagate on total loss not on invidiual ones.
            loss.backward()
            self.optimizer.step()

            # Now output all the log information to console and write the
            # necessary log values for TensorBoard.

            # Set the TensorBoard step.
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Update tracked metric and output to TensorBoard.
            self.train_metrics.update(
                self.metric.__class__.__name__, self.metric(output, target)
            )

            # For each specified logging to console step, show the current
            # epoch training information (batch progress, loss...). Usually
            # We don't show it every batch because there will be too many.
            if batch_idx % self.log_step == 0:

                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch,
                        self._progress(
                            batch_idx, self.data_loader, self.len_epoch
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

        # If there is a validation set, perform a validation step and update
        # the logged metrics.
        if self.validate:

            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        # Step learning rate if a scheduler is provided.
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch: int):
        """
        Single-epoch validation routine.

        Args:
            epoch: Current epoch number.

        Returns:
            Nothing.

        """

        # Set the model to evaluation mode and reset validation metrics.
        self.model.eval()
        self.valid_metrics.reset()

        # Fetch standardization and normalization factors.
        target_max = torch.tensor(self.validation_data_loader.target_max).to(
            self.device
        )
        target_min = torch.tensor(self.validation_data_loader.target_min).to(
            self.device
        )
        target_std = torch.tensor(self.validation_data_loader.target_std).to(
            self.device
        )
        target_mean = torch.tensor(self.validation_data_loader.target_mean).to(
            self.device
        )

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(
                self.validation_data_loader
            ):

                data, target = data.to(self.device), target.to(self.device)

                # Compute predictions.
                output = self.model(data)

                if self.validation_data_loader.normalize:
                    # De-normalize output and target for proper loss calculation.
                    output = output * (target_max - target_min) + target_min
                    target = target * (target_max - target_min) + target_min
                elif self.validation_data_loader.standardize:
                    # De-standardize output and target for proper loss calculation.
                    output = output * target_std + target_mean
                    target = target * target_std + target_mean

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
                        "loss{}".format(i), loss_i.item()
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
                    (epoch - 1) * len(self.validation_data_loader) + batch_idx,
                    "validation",
                )

        # Add histogram of model parameters to TensorBoard.
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()
