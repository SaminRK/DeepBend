from feature_model.helsep import HelSep

import pandas as pd
import numpy as np
import pytest

import unittest


class TestHelicalSeparationCounter(unittest.TestCase):
    def test_pair_dinc_dist_in(self):
        seq = "AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC"
        # Explanation
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        helsep = HelSep()
        all_dist = helsep._pair_dinc_dist_in(seq)
        pair_idx = helsep._dinc_pairs.index(("GC", "TT"))
        p_expected = np.bincount(
            [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31], minlength=49
        )[1:]

        self.assertListEqual(all_dist[pair_idx].tolist(), p_expected.tolist())

    def test_calculate_expected_p(self):
        df = HelSep().calculate_expected_p()
        self.assertEqual(df.shape, (136, 49))

    def test_helical_sep_of(self):
        seq = "AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC"
        # Explanation
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        # helical = max((0/n,0/n,0/n)) + max((0/n,0/n,0/n)) + max(1/n,0/n,1/n)
        # half-helical = max((1/n,1/n,1/n)) + max((0/n,1/n,0/n)) + max(0/n,0/n,0/n)
        # hs = h -hh

        helsep = HelSep()
        expected_dist = helsep.calculate_expected_p().values
        pair_idx = helsep._dinc_pairs.index(("GC", "TT"))

        # Normalize dist
        helical = (np.array([1, 0, 1]) / expected_dist[pair_idx, 28:31]).max()
        half_helical = (np.array([1, 1, 1]) / expected_dist[pair_idx, 3:6]).max() + (
            np.array([0, 1, 0]) / expected_dist[pair_idx, 13:16]
        ).max()

        hs = helsep.helical_sep_of([seq])
        print(hs.columns)
        assert hs.shape == (1, 137)
        assert hs["GC-TT"][0] == pytest.approx(helical - half_helical, abs=1e-3)
