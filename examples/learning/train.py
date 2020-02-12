#!/usr/bin/evn python3
# -*- coding: utf-8 -*-

""" Training script.

    This script carries out the training for a machine learning architecture
    using the specified run configuration file.

    Running the code:

        python3 train.py --h

        To obtain help about all the arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import argparse

import torch

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loaders as learning_loaders
import pypopsyn.learning.losses.losses as learning_losses
import pypopsyn.learning.metrics.metrics as learning_metrics
import pypopsyn.learning.models.models as learning_models
import pypopsyn.learning.trainers.trainer_basic as learning_trainer


def main(config):

    # Get handle for the logger ------------------------------------------------
    logger = config.get_logger("train")
    logger.info("Logger initialized...")

    # Setup data loaders -------------------------------------------------------
    logger.info("Creating data loaders...")
    loader = config.init_object("data_loader", learning_loaders)
    logger.info("Loader: {}".format(loader))

    # Build model --------------------------------------------------------------
    logger.info("Building model...")
    model = config.init_object("arch", learning_models)
    logger.info("Model architecture: {}".format(model))

    # Get handle for loss criterion --------------------------------------------
    logger.info("Creating loss criterion...")
    loss_criterion = config.init_object("loss", learning_losses)
    logger.info("Loss criterion: {}".format(loss_criterion))

    # Get handles for metric ---------------------------------------------------
    logger.info("Creating metrics...")
    metrics = config.init_object("metric", learning_metrics)
    logger.info("Metric: {}".format(metrics))
    # TODO: Handle multiple metrics.

    # Construct optimizer and scheduler ----------------------------------------
    trainable_parameters = filter(
        lambda p: p.requires_grad, model.parameters()
    )

    logger.info("Creating optimizer...")
    optimizer = config.init_object(
        "optimizer", torch.optim, trainable_parameters
    )
    logger.info("Optimizer {}".format(optimizer))

    logger.info("Creating scheduler...")
    scheduler = config.init_object(
        "lr_scheduler", torch.optim.lr_scheduler, optimizer
    )
    logger.info("Scheduler {}".format(scheduler))

    # Train the model ----------------------------------------------------------
    logger.info("Creating trainer...")
    trainer = learning_trainer.TrainerBasic(
        model,
        loss_criterion,
        metrics,
        optimizer,
        configuration=config,
        data_loader=loader,
        validation_data_loader=None,
        lr_scheduler=scheduler,
    )

    logger.info("{}".format(config))

    logger.info("Training model...")
    trainer.train()


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="PyPopSyn Population Synthesis Learning"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="config.json",
        help="Configuration file path",
    )

    configuration = configuration_parser.ConfigurationParser.from_args(args)

    main(configuration)
