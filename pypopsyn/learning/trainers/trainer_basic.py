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

    def _train_epoch(self, epoch: int):
        """
        Single-epoch training routine.

        Args:
            epoch: Current epoch number.

        Returns:
            Nothing.

        """

        # Set the model on training mode and reset all tracked metrics.
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

            # Update logged loss and metric.
            self.train_metrics.update("loss", loss.item())
            self.train_metrics.update(
                self.metric.__class__.__name__, self.metric(output, target)
            )

            # Output information to TensorBoard writer and to log file.
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

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

                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

            if batch_idx == self.len_epoch:
                break

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

                # Update validation metrics.
                self.valid_metrics.update("loss", loss.item())

                for metric in self.metric:
                    self.valid_metrics.update(
                        metric.__name__, metric(output, target)
                    )

                # Write output to Tensorboard and logger.
                self.writer.set_step(
                    (epoch - 1) * len(self.validation_data_loader) + batch_idx,
                    "validation",
                )

                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

        # Add histogram of model parameters to Tensorboard.
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()
