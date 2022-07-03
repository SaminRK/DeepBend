from util.kmer import KMer
from chromosome.chromosome import Chromosome


class TestKMer:
    def test_find_pos_w_rc(self):
        seq = "ATGCTATAGCCA"
        assert KMer.find_pos_w_rc("TA", seq).tolist() == [4, 5, 6, 7]
        assert KMer.find_pos_w_rc("TG", seq).tolist() == [1, 11]

    def test_find_pos(self):
        seq = "ATGCTATAGCCA"
        assert KMer.find_pos("TA", seq).tolist() == [4, 6]
        assert KMer.find_pos("TG", seq).tolist() == [1]

    def test_count(self, chrm_vl_mean7: Chromosome):
        assert KMer.count("TA", chrm_vl_mean7.seqf(25, 32)) == 1
        assert KMer.count("CG", chrm_vl_mean7.seqf(1, 10)) == 1
        assert KMer.count("TA", chrm_vl_mean7.seqf([6, 25], [13, 32])) == [0, 1]
        assert KMer.count("CG", chrm_vl_mean7.seqf([1, 5], [9, 8])) == [1, 0]
