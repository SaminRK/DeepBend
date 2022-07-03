from feature_model.correlation import Correlation
from feature_model.data_organizer import SequenceLibrary
from util.constants import RL, RL_LEN


class TestCorrelation:
    def test_kmer_corr(self):
        paths = Correlation.kmer_corr(SequenceLibrary(name=RL, quantity=RL_LEN))
        for p in paths:
            assert p.is_file()

    def test_helsep_corr(self):
        assert Correlation.helsep_corr(
            SequenceLibrary(name=RL, quantity=RL_LEN)
        ).is_file()
