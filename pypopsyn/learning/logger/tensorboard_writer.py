"""
    Tensorboard writer.

    This module provides a class for integrating Tensorboard logging functionality into your project,
    allowing for the visualization of metrics.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

"""

import datetime
import importlib


class TensorboardWriter:
    """
    A class for writing logs to Tensorboard, supporting both `torch.utils.tensorboard` and `tensorboardX`.

    Attributes:
        writer: The Tensorboard writer object.
        selected_module: The name of the selected module for writing logs.
        step: The current step for logging.
        mode: The current mode (e.g., "train" or "valid").
        tb_writer_ftns: A set of Tensorboard writing functions.
        tag_mode_exceptions: A set of functions that do not require mode tags.
        timer: A datetime object to track the time between steps.
    """

    def __init__(self, log_dir, logger, enabled):
        """
        Initializes the TensorboardWriter with the specified log directory, logger, and enable flag.

        Args:
            log_dir (str): The directory where Tensorboard logs will be saved.
            logger: The logger for displaying warnings or errors.
            enabled (bool): Flag to enable or disable Tensorboard logging.
        """

        self.writer = None
        self.selected_module = ""

        if enabled:

            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False

            for module in ["torch.utils.tensorboard", "tensorboardX"]:

                try:

                    self.writer = importlib.import_module(
                        module
                    ).SummaryWriter(log_dir)
                    succeeded = True
                    break

                except ImportError:

                    succeeded = False

                self.selected_module = module

            if not succeeded:

                message = (
                    "Warning: visualization (Tensorboard) is configured to use,"
                    " but currently not installed on this machine. Please "
                    " install TensorboardX with 'pip install tensorboardx', "
                    " upgrade PyTorch to version >= 1.1 to use "
                    " 'torch.utils.tensorboard' or turn off the option in the "
                    " JSON configuration file."
                )

                logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.datetime.now()

    def set_step(self, step, mode="train") -> None:
        """
        Sets the current step and mode, and logs the steps per second.

        Args:
            step (int): The current step for logging.
            mode (str): The current mode (default is "train").
        """

        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.datetime.now()
        else:
            duration = datetime.datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.datetime.now()

    def __getattr__(self, name):
        """
        Provides dynamic access to Tensorboard logging methods.

        Args:
            name (str): The name of the Tensorboard method to access.

        Returns:
            function: A wrapped function that adds additional information (step, tag) to the Tensorboard log entry.
            If visualization is configured to use returns add_data() methods of tensorboard with additional information
            (step, tag) added. Otherwise returns a blank function handle that does nothing.
        """

        if name in self.tb_writer_ftns:

            add_data = getattr(self.writer, name, None)

            def _wrapper(tag, data, *args, **kwargs):

                if add_data is not None:

                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)

                    add_data(tag, data, self.step, *args, **kwargs)

            return _wrapper

        else:

            try:

                attr = object.__getattr__(name)

            except AttributeError:

                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name
                    )
                )

            return attr
