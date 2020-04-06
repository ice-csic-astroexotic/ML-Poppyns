#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Inference script.

    This script infers a single sample from a dataset by leveraging a pretrained
    model and its architecture.

    Running the code:

        python3 infer.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse
import collections
import logging
import sys

import torch

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loaders as learning_loaders
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device


def infer(args, config):

    # Get handle for the logger ------------------------------------------------
    logger = config.get_logger("Inference")
    logger.info("Logger initialized...")

    # Setup data loaders -------------------------------------------------------
    logger.info("Creating data loaders...")
    loader = config.init_object("data_loader", learning_loaders)
    logger.info("Loader: {}".format(loader))

    # Build model --------------------------------------------------------------
    logger.info("Building model...")
    model = config.init_object("arch", learning_models)
    logger.info("Model architecture: {}".format(model))

    # Load pretrained model ----------------------------------------------------
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Prepare model for inference ----------------------------------------------
    logger.info("Preparing model for inference...")
    device, device_ids = request_device(logger, configuration["n_gpu"])
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    model.eval()

    # Select sample to infer and run inference ---------------------------------
    logger.info("Inferring sample {}...".format(args.sample))

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if i != args.sample:
                continue

            logger.info("Sample data {}...".format(data))
            logger.info("Sample labels {}...".format(target))

            data, target = data.to(device), target.to(device)
            output = model(data)

            logger.info(output)

    # Plot result and ground truth.
    # TODO.

    # Reproduce result with simulator.
    # TODO.


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters")

    parser.add_argument(
        "--configuration",
        type=str,
        default="examples/learning/config_NN1.json",
        help="Configuration file path.",
    )

    parser.add_argument("--resume", type=str, help="Path to pretrained model.")

    parser.add_argument(
        "--sample",
        nargs="?",
        type=int,
        default=0,
        help="Sample index in the dataset.",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["--dataset"], type=str, target=("data_loader;args;data_path")
        )
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        parser, options
    )

    infer(parser.parse_args(), configuration)
