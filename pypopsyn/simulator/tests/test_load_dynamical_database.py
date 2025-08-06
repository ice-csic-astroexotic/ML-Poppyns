"""
    Tests for the load_dynamical_database module.

        Authors:

            Michele Ronchi (ronchi@ice.csic.es)
"""
import json
import pathlib
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import pypopsyn.simulator.stellar_dynamics.coordinate_conversions as coco
import pypopsyn.simulator.stellar_dynamics.load_dynamical_database as ldyn
import utilities.samplers.memory_efficient_sampling as mes
from pypopsyn.simulator.config_simulator import cfg


@pytest.fixture()
def test_case_2():
    data = {
        "dyn_path": pathlib.Path("/mock/path/to/"),
        "dyn_data_path": pathlib.Path("/mock/path/to/final_pop_dyn.csv"),
        "dyn_config_path": pathlib.Path("/mock/path/to/configuration.json"),
        "n_batchsize": 10,
        "NS_number": 100,
        "idx_remove": [1, 3, 5],
        "mock_df_dyn": pd.DataFrame(
            {
                ("age", "[yr]"): [1e6, 2e6],
                ("r", "[kpc]"): [8.0, 8.5],
                ("phi", "[rad]"): [0.1, 0.2],
                ("z", "[kpc]"): [0.05, 0.1],
                ("v_r", "[km/s]"): [100, 200],
                ("v_phi", "[km/s]"): [220, 230],
                ("v_z", "[km/s]"): [10, 20],
            }
        ),
        "mock_cylindrical_to_all_coordinates": {
            "age": np.array([1e6, 2e6]),
            "r": np.array([8.0, 8.5]),
            "phi": np.array([0.1, 0.2]),
            "z": np.array([0.05, 0.1]),
            "v_r": np.array([100, 200]),
            "v_phi": np.array([220, 230]),
            "v_z": np.array([10, 20]),
            "ra": np.array([180.0, 181.0]),
            "dec": np.array([-30.0, -31.0]),
            "l": np.array([50.0, 51.0]),
            "b": np.array([10.0, 11.0]),
            "dist": np.array([8.0, 8.5]),
            "pm_ra": np.array([5.0, 5.1]),
            "pm_dec": np.array([-2.0, -2.1]),
            "v_ls": np.array([300.0, 310.0]),
            "idx": np.array([0, 1]),
        },
        "config_dyn": {
            "t_age_max": 1.0e8,
            "NS_number": 100,
            "kick_model": "kick_model",
            "sigma_k": 100.0,
            "vk_c": 100.0,
            "h_c": 100.0,
        },
        "expected_keys": {
            "age",
            "ra",
            "dec",
            "l",
            "b",
            "dist",
            "pm_ra",
            "pm_dec",
            "v_ls",
            "idx",
        },
        "cfg_expected": {
            "t_age_max": 1.0e8,
            "NS_number": 100,
            "kick_model": "kick_model",
            "sigma_k": 100.0,
            "vk_c": 100.0,
            "h_c": 100.0,
            "dyn_database_path": "/mock/path/to",
        },
    }

    return data


def test_load_database_dyn(monkeypatch, test_case_2):
    """
    Check that the dynamical database is correctly imported.
    """
    logger = MagicMock()

    # Save the original cfg to restore it later.
    original_cfg = cfg.copy()

    try:
        # Mock for dyn_data_path.exists() function.
        monkeypatch.setattr(
            pathlib.Path,
            "exists",
            lambda self: self == test_case_2["dyn_data_path"],
        )
        # Mock the configuration file loading.
        monkeypatch.setattr(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(test_case_2["config_dyn"])),
        )

        def mock_select(*args, **kwargs):
            return test_case_2["mock_df_dyn"]

        monkeypatch.setattr(mes, "select", mock_select)

        def mock_convert_cylindrical_to_all_coordinates(*args, **kwargs):
            return test_case_2["mock_cylindrical_to_all_coordinates"]

        monkeypatch.setattr(
            coco,
            "convert_cylindrical_to_all_coordinates",
            mock_convert_cylindrical_to_all_coordinates,
        )

        dict_out = ldyn.load_database_dyn(
            test_case_2["dyn_path"],
            test_case_2["n_batchsize"],
            test_case_2["idx_remove"],
            logger,
        )

        assert set(dict_out.keys()) == test_case_2["expected_keys"]
        for key in test_case_2["cfg_expected"].keys():
            assert key in cfg.keys()
            assert cfg[key] == test_case_2["cfg_expected"][key]

    finally:
        # Restore the original cfg in order for the other tests to work when pytest is run.
        cfg.clear()
        cfg.update(original_cfg)
