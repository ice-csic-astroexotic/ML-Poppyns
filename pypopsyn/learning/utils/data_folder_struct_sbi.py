"""
    Create data folder structure for running sbi.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import argparse
import os
from typing import Dict, Optional


def create_structure(base: str, structure: Dict[str, Optional[dict]]) -> None:
    """
    Recursively creates a fixed directory and file structure on the filesystem.

    Args:
        base (str): Base directory path where the folder structure will be created.
        structure (Dict[str, Optional[dict]]): Nested dictionary representing the folder/file structure.

    """
    for name, sub_structure in structure.items():
        path = os.path.join(base, name)
        os.makedirs(path, exist_ok=True)
        create_structure(path, sub_structure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a fixed directory structure for the SBI training pipeline."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base directory where the folder structure will be created.",
    )
    args = parser.parse_args()

    # Fixed folder structure (do not modify unless required for pipeline changes).
    folder_structure = {
        "data": {
            "test_dataset": {
                "generated_dataset": {"round_0": {}},
                "simulations": {},
            },
            "training_dataset": {
                "generated_dataset": {"round_0": {}},
                "simulations": {},
            },
        }
    }

    create_structure(args.base_path, folder_structure)
