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
import collections
import time
import typing

import numpy as np
import torch

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.initializers.initializers as learning_initializers
import pypopsyn.learning.loaders.loaders as learning_loaders
import pypopsyn.learning.losses.losses as learning_losses
import pypopsyn.learning.metrics.metrics as learning_metrics
import pypopsyn.learning.models.models as learning_models
import pypopsyn.learning.trainers.trainer_basic as learning_trainer
import pypopsyn.learning.utils.benchmark as benchmark


def main(config):

    trials = 1
    converged = False

    while (not converged) and (trials <= config["trials"]):

        # Get handle for the logger --------------------------------------------
        logger = config.get_logger("train")
        logger.info("Logger initialized...")

        # Show experiment information ------------------------------------------
        logger.info(
            "========================================================="
        )
        logger.info("Trial {} out of {}...".format(trials, config["trials"]))
        logger.info("Convergence thresholds:")
        for k, v in config["convergence"].items():
            logger.info("{}:{}".format(k, v))

        # Setup data loaders ---------------------------------------------------
        logger.info("Creating data loaders...")
        loader = config.init_object("data_loader", learning_loaders)
        logger.info("Loader: {}".format(loader))

        # Build model ----------------------------------------------------------
        logger.info("Building model...")
        model = config.init_object("arch", learning_models)
        logger.info("Model architecture: {}".format(model))

        # Initialize weights ---------------------------------------------------
        logger.info("Initializing weights...")
        weight_initializer = config.init_object(
            "weights_initializer", learning_initializers
        )
        logger.info("Weight initialization: {}".format(weight_initializer))
        # Apply the weight initialization scheme to every layer in the model.
        model.apply(weight_initializer)

        # Get handle for loss criterion ----------------------------------------
        logger.info("Creating loss criterion...")
        loss_criterion = config.init_object("loss", learning_losses)
        logger.info("Loss criterion: {}".format(loss_criterion))

        # Get handles for metric -----------------------------------------------
        logger.info("Creating metrics...")
        metric = config.init_object("metric", learning_metrics)
        logger.info("Metric: {}".format(metric))

        # Construct optimizer and scheduler ------------------------------------
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

        # Train the model ------------------------------------------------------
        logger.info("Creating trainer...")
        trainer = learning_trainer.TrainerBasic(
            model=model,
            criterion=loss_criterion,
            metric=metric,
            optimizer=optimizer,
            configuration=config,
            dataset_loader=loader,
            lr_scheduler=scheduler,
        )
        """
        # Benchmark model.
        logger.info(
            "Benchmarking model on device {}...".format(trainer.device)
        )
        # THIS GIVES PROBLEM SINCE loader IS NOT ITERABLE ANYMORE.
        input_dummy, labels_dummy = next(iter(loader))
        # TODO: Make sure this iter next does not skip the first batch next time.
        time_forward, time_backward = benchmark.benchmark(
            model, trainer.device, input_dummy, labels_dummy
        )
        logger.info("Forward pass time: {}[ms]".format(time_forward))
        logger.info("Backward pass time: {}[ms]".format(time_backward))
        """
        # Start training.
        logger.info("Training model...")
        train_results, best_result = trainer.train(trials)

        logger.info("Best losses: {}".format(train_results))
        logger.info("Best accuracies achieved: {}".format(best_result))

        # Iterate over the best individual train or val losses and check the
        # specified convergence criteria in the configuration file.
        converged = True
        for k, v in train_results.items():

            # If no convergence criteria is specified for a certain target, we
            # assume that is has converged.
            if (k + "_threshold") not in config["convergence"]:
                logger.info("No convergence criteria set for {}".format(k))
                continue

            logger.info(
                "Convergence threshold for {} is {}".format(
                    k, config["convergence"][(k + "_threshold")]
                )
            )

            if v < config["convergence"][(k + "_threshold")]:
                logger.info("Training converged for {}!".format(k))
            else:
                converged = False
                logger.info("Training did not converge for {}!".format(k))

        # If any of the targets has not converged, we will try to repeat the
        # training process.
        if converged:
            logger.info("Training has converged! Stopping.")
        else:
            logger.info("Training has not converged...")

        trials += 1


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="PyPopSyn Population Synthesis Learning"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="examples/learning/config_multiparameter.json",
        help="Configuration file path",
    )

    args.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training.",
    )

    CustomArgs = collections.namedtuple(
        "CustomArgs", "flags type nargs target"
    )

    options = [
        CustomArgs(
            ["--convergence"],
            type=float,
            nargs="?",
            target=("convergence_threshold"),
        ),
        CustomArgs(
            ["--dataset"],
            type=str,
            nargs="?",
            target=("data_loader;args;data_path"),
        ),
        CustomArgs(
            ["--initializer"],
            type=str,
            nargs="?",
            target=("weights_initializer;type"),
        ),
        CustomArgs(
            ["--ignored_inputs"],
            type=int,
            nargs="*",
            target=("data_loader;args;ignored_inputs"),
        ),
        CustomArgs(
            ["--ignored_labels"],
            type=int,
            nargs="*",
            target=("data_loader;args;ignored_labels"),
        ),
        CustomArgs(
            ["--batch_size"],
            type=int,
            nargs="?",
            target=("data_loader;args;batch_size"),
        ),
        CustomArgs(
            ["--input_shape"],
            type=int,
            nargs=3,
            target=("arch;args;input_shape"),
        ),
        CustomArgs(
            ["--num_parameters"],
            type=int,
            nargs="?",
            target=("arch;args;num_parameters"),
        ),
        CustomArgs(
            ["--save_dir"], type=str, nargs="?", target=("trainer;save_dir")
        ),
        CustomArgs(
            ["--normalize"],
            type=bool,
            nargs="?",
            target=("data_loader;args;normalize"),
        ),
        CustomArgs(
            ["--standardize"],
            type=bool,
            nargs="?",
            target=("data_loader;args;standardize"),
        ),
        CustomArgs(
            ["--lr"], type=float, nargs="?", target=("optimizer;args;lr")
        ),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        args, options
    )

    main(configuration)
