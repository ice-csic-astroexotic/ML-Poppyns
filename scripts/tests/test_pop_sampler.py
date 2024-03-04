"""
Test for the pop_sampler.py module.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

import scripts.pop_sampler as ps

TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "d": np.array([1.0, 2.0, 3.0, 4.0]),
        "expected_output": np.array(
            [0.67491214, 0.20467745, 0.0827621, 0.03764831]
        ),
    }

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "simulation_number": 3,
        "size": 20,
        "distance_cut": 5.0,
        "uniform": False,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "simulation_number": 3,
        "size": 20,
        "distance_cut": 5.0,
        "uniform": True,
    }

    return data


def test_calculate_selection_weights(test_case_1):
    """
    Testing if the selection weights are correctly calculated.
    """
    output = ps.calculate_selection_weights(test_case_1["d"])

    assert np.allclose(output, test_case_1["expected_output"], rtol=TOL)


@pytest.fixture
def simulation_dir(test_case_2, tmp_path):
    """
    Creating the structure inside the `simulated_populations` directory needed to test the `data_sampler`
    function.
        Args:
            tmp_path (pathlib.Path): temporary directory created by Pytest that will be deleted at the end of the test.
    """
    n_sim = test_case_2["simulation_number"]

    # Create a temporary directory for testing
    root_path = tmp_path / "simulated_populations"
    root_path.mkdir()

    # Create mock simulated population files
    mock_sim_dirs = [root_path / f"{i:06}" for i in range(n_sim)]

    for sim_dir in mock_sim_dirs:
        sim_dir.mkdir(parents=True)
        # Create a fake final_population.pkl.gz file
        header_final = pd.MultiIndex.from_arrays([["d"], ["[kpc]"]])
        df_pop = pd.DataFrame({"d": np.random.uniform(0, 10, 100)})
        df_pop.columns = header_final
        df_pop.to_pickle(
            sim_dir / "final_population.pkl.gz", compression="gzip"
        )
        # Create a fake override.json file
        override = {"sim_info": "info"}
        with open(sim_dir / "override.json", "w") as f:
            json.dump(override, f)

    return root_path


def test_data_sampler_with_weights(test_case_2, simulation_dir, tmp_path):
    """
    Testing that the simulated populations are correctly resampled with the weighted resampling.
        Args:
            tmp_path (pathlib.Path): temporary directory created by Pytest that will be deleted at the end of the test.
    """
    args = argparse.Namespace(
        data=str(simulation_dir),
        save_dir=str(tmp_path / "resampled_populations"),
        size=test_case_2["size"],
        distance_cut=test_case_2["distance_cut"],
        uniform=False,
    )

    ps.data_sampler(args)

    n_sim = test_case_2["simulation_number"]

    # Check if resampled population files are created
    resampled_dirs = list((tmp_path / "resampled_populations").glob("*"))
    assert len(resampled_dirs) == n_sim

    for resampled_dir in resampled_dirs:
        # Check if override.json file is copied
        assert (resampled_dir / "override.json").exists()

        # Check if the resampled final_population.pkl.gz file is created and has the correct number of stars
        df_resampled = pd.read_pickle(
            resampled_dir / "final_population.pkl.gz", compression="gzip"
        )
        assert len(df_resampled) == test_case_2["size"]

        # Check if distance_cut is applied correctly
        assert all(df_resampled["d"]["[kpc]"] <= test_case_2["distance_cut"])


def test_data_sampler_uniform(test_case_3, simulation_dir, tmp_path):
    """
    Testing that the simulated populations are correctly resampled with uniform resampling.
        Args:
            simulation_dir (pathlib.Path): temporary path to the mock simulated populations.
            tmp_path (pathlib.Path): temporary directory created by Pytest that will be deleted at the end of the test.
    """
    args = argparse.Namespace(
        data=str(simulation_dir),
        save_dir=str(tmp_path / "resampled_populations"),
        size=test_case_3["size"],
        distance_cut=test_case_3["distance_cut"],
        uniform=False,
    )

    ps.data_sampler(args)

    n_sim = test_case_3["simulation_number"]

    # Check if resampled population files are created
    resampled_dirs = list((tmp_path / "resampled_populations").glob("*"))
    assert len(resampled_dirs) == n_sim

    for resampled_dir in resampled_dirs:
        # Check if override.json file is copied
        assert (resampled_dir / "override.json").exists()

        # Check if the resampled final_population.pkl.gz file is created and has the correct number of stars
        df_resampled = pd.read_pickle(
            resampled_dir / "final_population.pkl.gz", compression="gzip"
        )
        assert len(df_resampled) == test_case_3["size"]

        # Check if distance_cut is applied correctly
        assert all(df_resampled["d"]["[kpc]"] <= test_case_3["distance_cut"])
