from feature_model.occurence import Occurence

import pandas as pd

import unittest


class TestOccurence(unittest.TestCase):
    def test_find_occurence(self):
        seq_list = ["AGTTC", "GATCC"]
        occur_dict = Occurence().find_occurence(seq_list, unit_size=2)
        expected = {
            "AA": 0,
            "AT": 1,
            "AG": 1,
            "AC": 0,
            "TA": 0,
            "TT": 1,
            "TG": 0,
            "TC": 2,
            "GA": 1,
            "GT": 1,
            "GG": 0,
            "GC": 0,
            "CA": 0,
            "CT": 0,
            "CG": 0,
            "CC": 1,
        }

        self.assertDictEqual(occur_dict, expected)

    def test_find_occurence_individual(self):
        df = pd.DataFrame({"Sequence": ["ACGT", "AAGT", "CTAG"]})
        df_occur = Occurence().find_occurence_individual(df, [2])

        assert len(df_occur) == 3
        assert len(df.columns) == 1
        assert len(df_occur.columns) == 1 + 4**2
        assert df_occur["AA"].tolist() == [0, 1, 0]
        assert df_occur["AG"].tolist() == [0, 1, 1]


if __name__ == "__main__":
    unittest.main()
