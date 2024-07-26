"""
    Inference script.

    This script infers a set of samples from a dataset by leveraging a
    pretrained model and its architecture.

    Display help message to run the code:

    python infer.py --h

    Displays all the relevant arguments that can be used.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import argparse
import collections
import pathlib

import pandas as pd
import torch

import pypopsyn.learning.configuration_parser as configuration_parser
import pypopsyn.learning.loaders.loaders as learning_loaders
import pypopsyn.learning.models.models as learning_models
from pypopsyn.learning.utils.request_device import request_device


def infer(
    args: argparse.Namespace, config: configuration_parser.ConfigurationParser
) -> None:
    """
    Perform inference on the model using the specified configuration and command line arguments.

    Args:
        args (Namespace): The command line arguments containing configuration options,
            save directory, and sample indices. It includes:
            - configuration (str): Path to the configuration file.
            - trained_model (str): Path to the pretrained model.
            - samples (List[int]): Sample indices in the dataset.
            - infer (bool): Flag to set up the inference saving path.
        config (ConfigurationParser): The configuration object containing settings for data loading,
            model architecture, and other parameters.
    """

    # Create the saving directory path.
    inference_results_path = config.log_dir
    pathlib.Path(inference_results_path).mkdir(parents=True, exist_ok=True)

    # Force data to load in a sequential manner without shuffling.
    config["test_data_loader"]["args"]["shuffle"] = False
    # Fix batch size to 1 for inference.
    config["test_data_loader"]["args"]["batch_size"] = 1

    # Get handle for the logger ------------------------------------------------
    logger = config.get_logger("Inference")
    logger.info("Logger initialized...")

    # Set up data loaders -------------------------------------------------------
    logger.info("Creating data loaders...")
    loader = config.init_object("test_data_loader", learning_loaders)
    logger.info(f"Loader: {loader}")
    # Fetch names of the target parameters to predict.
    target_names = loader.target_names

    # Build model --------------------------------------------------------------
    logger.info("Building model...")
    model = config.init_object("arch", learning_models)
    logger.info(f"Model architecture: {model}")

    # Prepare model for inference ----------------------------------------------
    logger.info("Preparing model for inference...")
    device, device_ids = request_device(logger, configuration["n_gpu"])
    if len(device_ids) >= 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    model.eval()

    # Load pretrained model ----------------------------------------------------
    logger.info(f"Loading checkpoint: {args.trained_model} ...")
    checkpoint = torch.load(args.trained_model)
    # Load the state dict
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Select sample to infer and run inference ---------------------------------
    logger.info(f"Inferring sample {args.samples}...")

    # Fetch standardization and normalization factors.
    target_max = torch.tensor(loader.target_max).to(device)
    target_min = torch.tensor(loader.target_min).to(device)
    target_std = torch.tensor(loader.target_std).to(device)
    target_mean = torch.tensor(loader.target_mean).to(device)

    target_values_dict = {}
    predicted_values_dict = {}

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if args.samples is not None and i not in args.samples:
                continue

            data, target = data.to(device), target.to(device)
            output = model(data)
            # De-normalize or de-standardize targets and outputs on the fly
            # if needed to rescale the loss values to a more readable range.
            # TODO: THIS COULD BE IMPROVED AND IDEALLY I WOULD LIKE THIS TO
            # BE DONE MORE TRANSPARENTLY, I DON'T KNOW HOW NOW.
            if loader.normalize:
                output = output * (target_max - target_min) + target_min
                target = target * (target_max - target_min) + target_min
            elif loader.standardize:
                output = output * target_std + target_mean
                target = target * target_std + target_mean

            logger.info(f"Sample labels {target}...")
            logger.info(f"Sample prediction {output}...")

            for j in range(len(output[0])):
                target_j = target[:, j]
                output_j = output[:, j]

                # Save target and output values into the partial dictionaries.
                target_values_dict.setdefault(
                    f"target:{target_names[j]}", []
                ).append(target_j.item())
                predicted_values_dict.setdefault(
                    f"predicted:{target_names[j]}", []
                ).append(output_j.item())

    # Merge the target and output dictionaries in a single dictionary.
    inference_results_dictionary = {
        **target_values_dict,
        **predicted_values_dict,
    }

    # Write the inference results dictionary into a .csv file.
    inference_results_filename = (
        f"{inference_results_path}/inference_results.csv"
    )

    df = pd.DataFrame(
        {
            key: pd.Series(value)
            for key, value in inference_results_dictionary.items()
        }
    )
    df.to_csv(inference_results_filename, encoding="utf-8", index=False)

    logger.info("File inference_results.csv generated.")

    # Reproduce result with simulator.
    # TODO.


if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="PyPopSyn Population Synthesis inferring"
    )

    args.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="pypopsyn/learning/config_multiparameter_MLP.json",
        help="Configuration file path.",
    )

    args.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="Path to pretrained model.",
    )

    args.add_argument(
        "--samples", nargs="*", type=int, help="Sample index in the dataset."
    )

    args.add_argument(
        "--infer",
        nargs="?",
        type=str,
        default=True,
        help="Flag to set up the inference saving path. If False you are in training mode.",
    )

    CustomArgs = collections.namedtuple(
        "CustomArgs", "flags type nargs target"
    )

    options = [
        CustomArgs(
            ["--dataset"],
            type=str,
            nargs="?",
            target="test_data_loader;args;dataset_path",
        ),
        CustomArgs(
            ["--dataset_statistics"],
            type=str,
            nargs="?",
            target="test_data_loader;args;statistic_path",
        ),
        CustomArgs(
            ["--filter_inputs"],
            type=int,
            nargs="*",
            target="test_data_loader;args;filter_inputs",
        ),
        CustomArgs(
            ["--filter_labels"],
            type=int,
            nargs="*",
            target="test_data_loader;args;filter_labels",
        ),
        CustomArgs(
            ["--input_shape"],
            type=int,
            nargs=3,
            target="arch;args;input_shape",
        ),
        CustomArgs(
            ["--num_parameters"],
            type=int,
            nargs="?",
            target="arch;args;num_parameters",
        ),
        CustomArgs(
            ["--save_dir"], type=str, nargs="?", target="infer;save_dir"
        ),
        CustomArgs(
            ["--normalize"],
            type=bool,
            nargs="?",
            target="test_data_loader;args;normalize",
        ),
        CustomArgs(
            ["--standardize"],
            type=bool,
            nargs="?",
            target="test_data_loader;args;standardize",
        ),
    ]

    configuration = configuration_parser.ConfigurationParser.from_args(
        args, options
    )

    infer(args.parse_args(), configuration)
