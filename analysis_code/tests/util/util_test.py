import unittest
from pathlib import Path
import os

import pandas as pd
import numpy as np

from util.util import (
    NumpyTool,
    PlotUtil,
    get_possible_seq,
    cut_sequence,
    rev_comp,
    append_reverse_compliment,
    FileSave,
    PathObtain,
)

from util.constants import GDataSubDir

# https://stackoverflow.com/a/31832447/7283201


class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_reverse_compliment_of(self):
        res = rev_comp("ATGCTAAC")
        assert res == "GTTAGCAT"

    def test_append_reverse_compliment(self):
        df = pd.DataFrame({"Sequence": ["ATGCCGT", "GCGATGC"], "Col2": [5, 6]})
        rdf = append_reverse_compliment(df)

        self.assertGreater(len(rdf), len(df))
        self.assertCountEqual(
            rdf["Sequence"].tolist(), ["ATGCCGT", "GCGATGC", "ACGGCAT", "GCATCGC"]
        )
        self.assertCountEqual(rdf["Col2"].tolist(), [5, 6, 5, 6])

    def test_get_possible_seq_two(self):
        possib_seq = get_possible_seq(size=2)
        expected = [
            "AA",
            "AT",
            "AG",
            "AC",
            "TA",
            "TT",
            "TG",
            "TC",
            "GA",
            "GT",
            "GG",
            "GC",
            "CA",
            "CT",
            "CG",
            "CC",
        ]

        # Test two list have same content without regard to their order
        self.assertCountEqual(possib_seq, expected)

    def test_cut_sequence(self):
        df = pd.DataFrame({"Sequence": ["abcde", "fghij"]})
        cdf = cut_sequence(df, 2, 4)
        cdf_seq_list = cdf["Sequence"].tolist()
        expected = ["bcd", "ghi"]

        # Test two list have same content, also regarding their order
        self.assertListEqual(cdf_seq_list, expected)


class TestFileSave:
    def test_fasta(self):
        assert FileSave.fasta(
            ["ATC", "GTAA"],
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.TEST}/test.fasta",
        ).is_file()

    def test_append_tsv(self):
        sample_df = pd.DataFrame({"a": [1, 2], "b": [11, 12], "c": [21, 25]})
        path = Path("data/generated_data/test_append.tsv")

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)

        sample_df.to_csv(path, sep="\t", index=False)
        other_df = pd.DataFrame({"a": [3], "b": [13], "d": [31]})

        FileSave.append_tsv(other_df, path)
        append_df = pd.read_csv(path, sep="\t")
        assert len(append_df) == 3
        assert len(append_df.columns) == 4
        os.remove(path)


class TestPlotUtil:
    def test_bar_stacked(self):
        series_labels = ["Series 1", "Series 2"]

        data = [[0.2, 0.3, 0.35, 0.3], [0.8, 0.7, 0.6, 0.5]]

        category_labels = ["Cat A", "Cat B", "Cat C", "Cat D"]

        PlotUtil.bar_stacked(
            data,
            series_labels,
            category_labels,
            show_values=True,
            value_format="{:.1f}",
            colors=["tab:orange", "tab:green"],
            y_label="Quantity (units)",
        )
        FileSave.figure(f"{PathObtain.figure_dir()}/test/stacked_bar.png")
        assert True


class TestNumpyTool:
    def test_match_pattern(self):
        container = np.array([True, True, False, False, True, False, False, True, True])
        assert NumpyTool.match_pattern(container, np.array([False, True])).tolist() == [3, 6]
        assert NumpyTool.match_pattern(container, [True, False, False]).tolist() == [1, 4]


if __name__ == "__main__":
    unittest.main()
