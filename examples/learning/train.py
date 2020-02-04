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

import pypopsyn.learning.configuration_parser as configuration_parser


def main(config):

    # Get handle for the logger ------------------------------------------------
    logger = config.get_logger("train")
    logger.info("Logger initialized...")

    # Setup data loaders -------------------------------------------------------
    # TODO.

    # Build model --------------------------------------------------------------
    # TODO.

    # Get handles for loss criterion -------------------------------------------
    # TODO.

    # Get handle for metric ----------------------------------------------------
    # TODO.

    # Construct optimizer and scheduler ----------------------------------------
    # TODO.

    # Train the model ----------------------------------------------------------
    # TODO.


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
