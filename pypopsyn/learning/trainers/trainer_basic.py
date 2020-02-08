import numpy as np
import torch
from torchvision.utils import make_grid

import pypopsyn.learning.utils as learning_utils
import pypopsyn.learning.utils.inf_loop
import pypopsyn.learning.utils.metric_tracker

from .trainer_base import BaseTrainer


class TrainerBasic(BaseTrainer):
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
        len_epoch=None,
    ):

        super().__init__(model, criterion, metrics, optimizer, configuration)

        self.configuration = configuration
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = learning_utils.inf_loop.inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.validation_data_loader = validation_data_loader
        self.validate = self.validation_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = learning_utils.metric_tracker.MetricTracker(
            "loss", *[self.metrics.__class__.__name__], writer=self.writer
        )
        self.valid_metrics = learning_utils.metric_tracker.MetricTracker(
            "loss", *[self.metrics.__class__.__name__], writer=self.writer
        )

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            self.train_metrics.update(
                self.metrics.__class__.__name__, self.metrics(output, target)
            )

            if batch_idx % self.log_step == 0:

                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.validate:

            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):

        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(
                self.validation_data_loader
            ):

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.validation_data_loader) + batch_idx,
                    "validation",
                )
                self.valid_metrics.update("loss", loss.item())

                for metric in self.metrics:
                    self.valid_metrics.update(
                        metric.__name__, metric(output, target)
                    )

                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):

        base = "[{}/{} ({:.0f}%)]"

        if hasattr(self.data_loader, "n_samples"):

            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples

        else:

            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)
