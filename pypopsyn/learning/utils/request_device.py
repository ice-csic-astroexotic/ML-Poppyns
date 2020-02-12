""" Request Device.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import typing

import torch


def request_device(
    logger, num_gpu: int = 0
) -> typing.Tuple[torch.device, list]:
    """
    Selects the requested devices for training/testing.

    Args
        logger: A logger to log information to.
        num_gpu: Number of GPUs requested.

    Returns:
        A tuple containing the kind of device the pipeline can run on and
        a list of devices if available.

        If no GPUs are avaible or zero are requested, the returned device
        is CPU.

    """

    num_available_gpus = torch.cuda.device_count()

    # Check if GPUs are requested but no GPUs are available.
    if num_gpu > 0 and num_available_gpus == 0:

        logger.warning(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        num_gpu = 0

    # Check if the number of requested GPUs exceeds the available.
    if num_gpu > num_available_gpus:

        logger.warning(
            "Warning: The number of GPU's configured to use is {} "
            "but only {} are available "
            "on this machine.".format(num_gpu, num_available_gpus)
        )
        num_gpu = num_available_gpus

    device = torch.device("cuda:0" if num_gpu > 0 else "cpu")
    list_ids = list(range(num_gpu))

    return device, list_ids
