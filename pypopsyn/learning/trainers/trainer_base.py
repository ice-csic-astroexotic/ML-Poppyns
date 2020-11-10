""" Base Trainer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import json
import pathlib
import typing
from abc import abstractmethod

import numpy as np
import torch
from numpy import inf

from pypopsyn.learning.configuration_parser import ConfigurationParser
from pypopsyn.learning.logger.tensorboard_writer import TensorboardWriter
from pypopsyn.learning.utils.request_device import request_device


class BaseTrainer:
    """
    Base trainer.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        metric,
        optimizer,
        configuration: ConfigurationParser,
    ):
        """
        Trainer initialization.

        Args:
            model: Network to train.
            criterion: Loss criterion.
            metric: Metric to validate.
            optimizer: Optimizer for training.
            configuration: Current experiment configuration.

        Returns:
            Nothing.

        """

        self.configuration = configuration

        self.logger = configuration.get_logger(
            "trainer", configuration["trainer"]["verbosity"]
        )

        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer

        # Setup GPU device if available, move model into configured device.
        self.logger.info(
            "Requesting {} GPUs...".format(configuration["n_gpu"])
        )
        self.device, device_ids = request_device(
            self.logger, configuration["n_gpu"]
        )
        self.logger.info("Devices obtained: {}".format(device_ids))
        self.model = model.to(self.device)
        if len(device_ids) >= 1:
            self.logger.info(
                "{} GPU detected, running in parallel!".format(len(device_ids))
            )
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Trainer configuration and parameter fetching from config dictionary.
        self.logger.info("Configuring trainer...")
        trainer_configuration = configuration["trainer"]
        self.epochs = trainer_configuration["epochs"]
        self.save_period = trainer_configuration["save_period"]
        self.monitor_best = self.metric.initial_value()
        self.early_stop = trainer_configuration.get("early_stop", inf)
        self.start_epoch = 1
        self.checkpoint_dir = self.configuration.save_dir
        self.log_dir = self.configuration.log_dir

        # Initialize training and validation JSONs.
        self.train_json_path = pathlib.Path().joinpath(
            self.log_dir, "train_result.json"
        )
        self.train_json: dict = {}
        with open(self.train_json_path, "w") as f:
            json.dump(self.train_json, f, indent=2, sort_keys=True)

        self.train_eval_json_path = pathlib.Path().joinpath(
            self.log_dir, "train_eval_result.json"
        )
        self.train_eval_json: dict = {}
        with open(self.train_eval_json_path, "w") as f:
            json.dump(self.train_eval_json, f, indent=2, sort_keys=True)

        self.validation_json_path = pathlib.Path().joinpath(
            self.log_dir, "validation_result.json"
        )
        self.validation_json: dict = {}
        with open(self.validation_json_path, "w") as f:
            json.dump(self.validation_json, f, indent=2, sort_keys=True)

        # setup visualization writer instance
        self.writer = TensorboardWriter(
            configuration.log_dir,
            self.logger,
            trainer_configuration["tensorboard"],
        )

        if configuration.resume is not None:
            self._resume_checkpoint(configuration.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self, trial: int = None) -> typing.Tuple[dict, float]:

        """
        Main training procedure.

        This captures the whole training process. It calls the specific epoch
        training method from the derived trainers and updates the logging
        information accordingly.

        Furthermore, it also monitors the metric to check if it has improved
        or not and perform early stopping if needed.

        At last, it checkpoints the training process at the specified interval;
        it also saves the most accurate model to `best_model.pth`.

        Args:
            trial (int): the current trial to add suffixes to the saved models
                and checkpoints. Can be none if no trial is specified.

        Returns:
            dict: a dictionary with the best values for each individual loss for
            each one of the targets.
            float: the best result for the specified metric over the whole
            training process (validation accuracy according to the metric if
            validation is performed and training accuracy otherwise).

        """

        best_losses = {}
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):

            epoch_str = f"{epoch:05d}"

            self.logger.info(
                "************************************************"
            )
            self.logger.info("Epoch {}".format(epoch_str))
            self.logger.info("Best accuracy: {}".format(self.monitor_best))

            # Run one epoch and fetch the result dictionaries for train/val and
            # the losses that will be used for convergence.
            (
                train_result,
                val_result,
                train_eval_result,
                losses,
            ) = self._train_epoch(epoch)

            # Update current epoch logging dictionary with the results from the
            # training epoch (usually loss and accuracy averages).
            log = {"epoch": epoch}
            log.update(train_result)
            current_result = log[self.metric.__class__.__name__]

            # Print training per-epoch logged information to the screen.
            self.logger.info("Training results...")
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # Log results to training JSON.
            self.train_json[epoch_str] = {}
            for key, value in log.items():
                if key == "epoch":
                    continue
                self.train_json[epoch_str][key] = value

            with open(self.train_json_path, "w") as f:
                json.dump(self.train_json, f, indent=2, sort_keys=True)

            # Update current epoch logging dictionary with the results from the
            # training evaluation epoch (usually loss and accuracy averages).
            train_eval_log = {"epoch": epoch}
            train_eval_log.update(train_eval_result)

            # Print training per-epoch logged information to the screen.
            self.logger.info("Training evaluation results...")
            for key, value in train_eval_log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # Log results to training JSON.
            self.train_eval_json[epoch_str] = {}
            for key, value in train_eval_log.items():
                if key == "epoch":
                    continue
                self.train_eval_json[epoch_str][key] = value

            with open(self.train_eval_json_path, "w") as f:
                json.dump(self.train_eval_json, f, indent=2, sort_keys=True)

            # Print validation information if validation was performed and use
            # it to update the training tracking metrics if so (like the current
            # best loss so far).
            if val_result is not None:

                # Update current epoch logging dictionary with the results from
                # the validation epoch (usually loss and accuracy averages).
                val_log = {"epoch": epoch}
                val_log.update(val_result)
                current_result = val_log[self.metric.__class__.__name__]

                # Print validation per-epoch logged information to the screen.
                self.logger.info("Validation results...")
                for key, value in val_log.items():
                    self.logger.info("    {:15s}: {}".format(str(key), value))

                # Log results to validation JSON.
                self.validation_json[epoch_str] = {}
                for key, value in val_log.items():
                    if key == "epoch":
                        continue
                    self.validation_json[epoch_str][key] = value

                with open(self.validation_json_path, "w") as f:
                    json.dump(
                        self.validation_json, f, indent=2, sort_keys=True
                    )

            # Check whether model performance improved or not, according
            # to specified metric behavior (minimum or maximum). The metric will
            # be the validation one if validation is performed or training if
            # no validation is carried out.
            best = False

            if self.metric.improved(self.monitor_best, current_result):
                # The current result improves the running best, save it and
                # reset the patience counter for early stopping.
                self.monitor_best = current_result
                not_improved_count = 0
                best = True
                best_losses = losses
                self.logger.info("Metric improved!")
            else:
                # The current result did not improve the running best, increase
                # the patience counter for early stopping.
                self.logger.info(
                    "Metric did not improve for {} epochs...".format(
                        not_improved_count
                    )
                )
                not_improved_count += 1

            # Perform early stopping if the metric has not improved for
            # the specified number of epochs (patience).
            if not_improved_count > self.early_stop:
                self.logger.info(
                    "Target metric did not improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                break

            # Create checkpoint at the requested interval.
            if (epoch % self.save_period) == 0:
                self._save_checkpoint(
                    epoch,
                    "checkpoint_trial{}_epoch{}.pth".format(trial, epoch),
                )
                self.logger.info("Saved checkpoint...")

            # Save best model if it is the case.
            if best:
                self._save_checkpoint(
                    epoch, "best_model_trial{}.pth".format(trial)
                )
                self.logger.info("Saved best model so far...")

        return best_losses, self.monitor_best

    def _progress(self, batch_idx: int, data_loader, len_epoch: int) -> str:
        """
        Epoch progress tracker.

        Args:
            batch_idx: Current batch index.
            data_loader: Data loader in use.
            len_epoch: Length of a whole epoch.

        Returns:
            A string representation of the progress in the current epoch.

        """

        base = "[{}/{} ({:.0f}%)]"

        if hasattr(data_loader, "n_samples"):

            current = (batch_idx + 1) * data_loader.batch_size
            total = data_loader.n_samples

        else:

            current = batch_idx + 1
            total = len_epoch

        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        """
        Checkpoint saving.

        Saves the current state of the training process to a checkpoint:
        the model architecture, the epoch, the optimizer state, and the
        configuration.

        Args:
            epoch: current training epoch.
            filename: filename to save the checkpoint to.

        Returns:
            Nothing. Saves the checkpoint in the checkpoint folder with
            the specified filename.

        """

        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.monitor_best,
            "config": self.configuration,
        }

        filename = str(self.checkpoint_dir / filename)
        torch.save(state, filename)

    def _resume_checkpoint(self, checkpoint_path) -> None:
        """
        Resumes a checkpoint to continue training.

        Checks if the checkpoint architecture matches the current architecture,
        if not, no parameters are loaded. It also performs the same check for
        the optimizer state.

        Args:
            checkpoint_path: Path to checkpoint to resume.

        Returns:
            Nothing.

        """

        checkpoint_path = str(checkpoint_path)

        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        self.start_epoch = checkpoint["epoch"] + 1
        self.monitor_best = checkpoint["monitor_best"]

        # Only load model parameters (architecture) if the checkpoint arch
        # is the same as the current architecture.
        if checkpoint["config"]["arch"] != self.configuration["arch"]:

            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint."
            )

        else:

            self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state from checkpoint only when the optimizer type
        # from the checkpoint is the same as the current in use.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.configuration["optimizer"]["type"]
        ):

            self.logger.warning(
                "Optimizer type given in config file is different from that of "
                "checkpoint. Optimizer parameters not being resumed."
            )

        else:

            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch
            )
        )
