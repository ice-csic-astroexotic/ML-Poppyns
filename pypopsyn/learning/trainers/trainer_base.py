""" Base Trainer.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

from abc import abstractmethod

import torch
from numpy import inf

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
        configuration: dict,
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
        self.device, device_ids = request_device(configuration["n_gpu"])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        trainer_configuration = configuration["trainer"]
        self.epochs = trainer_configuration["epochs"]
        self.save_period = trainer_configuration["save_period"]
        self.monitor_best = self.metric.initial_value()
        self.early_stop = trainer_configuration.get("early_stop", inf)
        self.start_epoch = 1
        self.checkpoint_dir = configuration.save_dir

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

    def train(self) -> None:

        """
        Main training procedure.

        This captures the whole training process. It calls the specific epoch
        training method from the derived trainers and updates the logging
        information accordingly.

        Furthermore, it also monitors the metric to check if it has improved
        or not and perform early stopping if needed.

        At last, it checkpoints the training process at the specified interval.

        Args:
            None.

        Returns:
            Nothing.

        """

        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_epoch(epoch)

            # Save logged informations into logging dictionary.
            log = {"epoch": epoch}
            log.update(result)

            # Print logged information to the screen.
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # Evaluate model performance according to configured metric.
            # Save best checkpoint as model_best.
            best = False

            # Check whether model performance improved or not, according
            # to specified metric behavior (minimum or maximum).
            if self.metric.improved(
                self.monitor_best, log[self.metric.__class__.__name__]
            ):

                self.current_best = log[self.metric.__class__.__name__]
                not_improved_count = 0
                best = True

            else:
                not_improved_count += 1

            # Perform early stopping if the metric has not improved for
            # the specified number of epochs.
            if not_improved_count > self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                break

            # Create checkpoint.
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

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

            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples

        else:

            current = batch_idx
            total = len_epoch

        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Checkpoint saving.

        Saves the current state of the training process to a checkpoint:
        the model architecture, the epoch, the optimizer state, and the
        configuration.

        Args:
            epoch: Current epoch number.
            save_best: Whether or not this is the best model so far.

        Returns:
            Nothing. Saves the checkpoint in the checkpoint folder using
            the epoch number as suffix. If the current epoch has produced
            the best model so far, it is saved as the best model.

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

        filename = str(
            self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch)
        )

        torch.save(state, filename)

        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:

            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

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
