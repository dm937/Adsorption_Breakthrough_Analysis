# -*- coding: utf-8 -*-

"""Trivial version test."""

import copy
import os
import pickle
import unittest

import pandas as pd

from AdsorptionBreakthroughAnalysis import ExperimentalSetup, experiment_analysis


class TestVersion(unittest.TestCase):
    """Trivially test a version."""

    def _load_pybreak_obj(self, ExperimentalSetup, root_path, blank):
        save_path = "tests/data/processed_data/"

        if blank == True:
            fpath = os.path.join(save_path, "bench_loading_experiment_22perc_ZIF8_blank.pickle")
            coriolis_file_name = os.path.join(
                root_path, "220819-ZIF-8-dry-22%and60%40C20220819134938.txt"
            )
            MS_file_name = os.path.join(root_path, "220819ZIF8Blank22and60%CO2at40C.csv")
            sorted_csv = pd.read_csv(
                os.path.join(
                    save_path,
                    "bench_220819-ZIF-8-dry-22%and60%40C20220819134938.txt_220819ZIF8Blank22and60%CO2at40C.csv.csv",
                ),
                index_col=0,
            )

        else:
            fpath = os.path.join(save_path, "bench_loading_experiment_22perc_ZIF8.pickle")
            coriolis_file_name = os.path.join(root_path, "220824-ZIF-8-4and22%at40C.txt")
            MS_file_name = os.path.join(root_path, "220824-ZIF-8-4and22%at40C.csv")
            sorted_csv = pd.read_csv(
                os.path.join(
                    save_path,
                    "bench_220824-ZIF-8-4and22%at40C.txt_220824-ZIF-8-4and22%at40C.csv.csv",
                ),
                index_col=0,
            )

        pybreak_obj = experiment_analysis(
            coriolis_file_name=coriolis_file_name,
            MS_file_name=MS_file_name,
            conditions=copy.deepcopy(ExperimentalSetup),
        )

        with open(fpath, "rb") as f:
            loading = pickle.load(f)

        return pybreak_obj, loading, sorted_csv

    def test_sorted_data_blank(self):
        """Test the sorted data is the same as before."""
        ExperimentalSetup["breakthrough_start"] = 3600 * 0 + 60 * 45 + 30
        ExperimentalSetup["breakthrough_end"] = 3600 * 1 + 60 * 4 + 30
        ExperimentalSetup["LowConcCo2"] = False
        ExperimentalSetup["Coriolis_start"] = 70
        root_path = "tests/data/experimental_data"

        blank = True
        pybreak_obj, loading, sorted_csv = self._load_pybreak_obj(
            ExperimentalSetup, root_path, blank
        )

        for key, value in pybreak_obj.calculate_loading()[0][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[0][1][key], loading[0][1][key], places=1
            )

        for key, value in pybreak_obj.calculate_loading()[1][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[1][1][key], loading[1][1][key], places=1
            )

        bench_sorted_dict = sorted_csv.describe().T[["mean", "count", "std"]].to_dict()
        sorted_dict_compare = (
            pybreak_obj.sorted_data.describe().T[["mean", "count", "std"]].to_dict()
        )

        for key, value in bench_sorted_dict["mean"].items():
            self.assertAlmostEqual(
                sorted_dict_compare["mean"][key], bench_sorted_dict["mean"][key], places=1
            )
            self.assertEqual(sorted_dict_compare["count"][key], bench_sorted_dict["count"][key])
            self.assertAlmostEqual(
                sorted_dict_compare["std"][key], bench_sorted_dict["std"][key], places=1
            )

        # pybreak_obj_sorted_data = pd.read_csv(f"tests/data/processed_data/bench_{coriolis_file_name.split('/')[-1]}_{MS_file_name.split('/')[-1]}.csv")

    def test_sorted_data(self):
        """Test the sorted data is the same as before."""
        ExperimentalSetup["T_exp"] = 313
        ExperimentalSetup["breakthrough_start"] = 3600 * 5 + 60 * 1 + 40
        ExperimentalSetup["breakthrough_end"] = 3600 * 5 + 60 * 23 + 10
        ExperimentalSetup["LowConcCo2"] = False
        ExperimentalSetup["Coriolis_start"] = 30
        root_path = "tests/data/experimental_data"

        blank = False
        pybreak_obj, loading, sorted_csv = self._load_pybreak_obj(
            ExperimentalSetup, root_path, blank
        )

        for key, value in pybreak_obj.calculate_loading()[0][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[0][1][key], loading[0][1][key], places=1
            )

        for key, value in pybreak_obj.calculate_loading()[1][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[1][1][key], loading[1][1][key], places=1
            )

        for key, value in pybreak_obj.calculate_loading()[0][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[0][1][key], loading[0][1][key], places=1
            )

        for key, value in pybreak_obj.calculate_loading()[1][1].items():
            self.assertAlmostEqual(
                pybreak_obj.calculate_loading()[1][1][key], loading[1][1][key], places=1
            )

        # pybreak_obj_sorted_data = pd.read_csv(f"tests/data/processed_data/bench_{coriolis_file_name.split('/')[-1]}_{MS_file_name.split('/')[-1]}.csv")
