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
            "loss", *[self.metric.__class__.__name__], writer=self.writer
        )
        self.valid_metrics = learning_utils.metric_tracker.MetricTracker(
            "loss", *[self.metric.__class__.__name__], writer=self.writer
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

            # Training step: zero gradients, compute predictions, calculate
            # loss and perform backward pass.
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Now output all the log information to console and write the
            # necessary log values for TensorBoard.

            # Set the TensorBoard step.
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Update tracked loss and output to TensorBoard.
            self.train_metrics.update("loss", loss.item())
            # Update tracked metric and output to TensorBoard.
            self.train_metrics.update(
                self.metric.__class__.__name__, self.metric(output, target)
            )
            # Show the input images of this batch on TensorBoard.
            # TODO: temporarily disabled until we find a better way to
            # represent arbitrary channel images.
            # self.writer.add_image(
            #    "input", make_grid(data.cpu(), nrow=8, normalize=True)
            # )

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

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(
                self.validation_data_loader
            ):

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                # Update tracked loss and output to TensorBoard.
                self.valid_metrics.update("loss", loss.item())
                # Update tracked metric and output to TensorBoard.
                self.valid_metrics.update(
                    self.metric.__class__.__name__, self.metric(output, target)
                )

                # Set tensorboard step.
                self.writer.set_step(
                    (epoch - 1) * len(self.validation_data_loader) + batch_idx,
                    "validation",
                )

        # Add histogram of model parameters to Tensorboard.
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()
