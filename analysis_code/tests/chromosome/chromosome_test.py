import unittest
import time

import numpy as np
import pytest

from chromosome.chromosome import Chromosome, ChrmCalc, Spread
from util.reader import DNASequenceReader
from util.constants import CHRVL_LEN, CHRV_TOTAL_BP


class TestChrmCalc(unittest.TestCase):
    def test_moving_avg(self):
        arr = np.array([4, 6, 1, -9, 2, 7, 3])
        ma = ChrmCalc.moving_avg(arr, 4)
        assert ma.tolist() == [0.5, 0, 0.25, 0.75]

    def test_get_total_bp(self):
        assert ChrmCalc.total_bp(5) == 78


class TestSpread:
    def test_covering_sequences_at(self, chrm_vl):
        spread = Spread(chrm_vl._df["C0"].to_numpy(), chrm_vl.id)
        arr = spread._covering_sequences_at(30)
        assert arr.tolist() == [1, 2, 3, 4, 5]
        arr = spread._covering_sequences_at(485)
        assert arr.tolist() == [64, 65, 66, 67, 68, 69, 70]
        arr = spread._covering_sequences_at(576860)
        assert arr.tolist() == [82403, 82404]

    def test_mean_of_7(self, chrm_vl: Chromosome):
        spread = Spread(chrm_vl._df["C0"].to_numpy(), chrm_vl.id)
        spread_c0 = spread._mean_of_7()
        assert spread_c0.shape == (CHRV_TOTAL_BP,)
        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]

        assert np.all(samples < 2.5)
        assert np.all(samples > -2.5)

    def test_mean_of_covering_seq(self, chrm_vl: Chromosome):
        spread = Spread(chrm_vl._df["C0"].to_numpy(), chrm_vl.id)
        spread_c0 = spread._mean_of_covering_seq()
        assert spread_c0.shape == (CHRV_TOTAL_BP,)
        assert spread_c0[48:51] == pytest.approx([-0.11, -0.1641, -0.1777], abs=1e-4)

        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]
        assert np.all(samples < 2.5)
        assert np.all(samples > -2.5)

    def test_spread_c0_weighted(self, chrm_vl: Chromosome):
        spread = Spread(chrm_vl._df["C0"].to_numpy(), chrm_vl.id)
        spread_c0 = spread._weighted_covering_seq()
        assert spread_c0.shape == (CHRV_TOTAL_BP,)

        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]
        assert np.all(samples < 2.5)
        assert np.all(samples > -2.5)


class TestChromosome:
    def test_get_chr_prediction(self, chrm_i: Chromosome):
        predict_df = chrm_i._get_chrm_df()
        assert predict_df.columns.tolist() == ["Sequence #", "Sequence", "C0"]

    def test_seq(self, chrm_vl: Chromosome):
        seq = chrm_vl.seq
        assert len(seq) == CHRV_TOTAL_BP
        assert (
            seq[60000:60100] == DNASequenceReader.read_yeast_genome_file(5)[60000:60100]
        )

    def test_seqf(self, chrm_vl: Chromosome):
        assert chrm_vl.seqf(100, 105) == "CTACTC"
        assert chrm_vl.seqf([100, 200], [105, 205]) == ["CTACTC", "AATTTC"]

    def test_mean_c0_around_bps(self, chrm_i: Chromosome):
        mean = chrm_i.mean_c0_around_bps([5000, 10000, 12000], 60, 40)
        assert mean.shape == (60 + 40 + 1,)

    def test_mean_c0_of_segments(self, chrm_vl: Chromosome):
        mn = chrm_vl.mean_c0_of_segments([5000, 8000], 100, 50)
        assert -2 < mn < 2

    def test_mean_c0_at_bps(self, chrm_i: Chromosome):
        mn = chrm_i.mean_c0_at_bps([12000, 20000], 200, 200)
        assert mn[0] > -1
        assert mn[0] < 1
        assert mn[1] > -1
        assert mn[1] < 1

    def test_c0_spread_saving(self, chrm_i: Chromosome):
        t = time.time()
        sp_one = chrm_i.c0_spread()
        dr_one = time.time() - t
        assert hasattr(chrm_i, "_c0_spread")

        t = time.time()
        sp_two = chrm_i.c0_spread()
        assert len(sp_one) == len(sp_two)

        dr_two = time.time() - t
        assert dr_two < dr_one * 0.01
