import numpy as np
import pytest

from motif.motifs import MotifsM30, MotifsM35, LEN_MOTIF, PlotMotifs
from chromosome.regions import Regions
from util.reader import DNASequenceReader
from util.constants import YeastChrNumList, FigSubDir
from util.util import roman_to_num


@pytest.fixture
def motifsm35():
    return MotifsM35()


class TestMotifsM35:
    def test_motif40_score(self, motifsm35: MotifsM35):
        motif_40 = "_GAAGAGC"
        seq = DNASequenceReader.read_yeast_genome_file(roman_to_num(YeastChrNumList[4]))
        match_pos = seq.find(motif_40[1:])
        max_score_pos = np.where(motifsm35._running_score[40] > 14)[0][0]
        assert max_score_pos - LEN_MOTIF / 2 + 1 == match_pos

    def test_enrichment(self, motifsm35: MotifsM35, rgns_simp_vl: Regions):
        assert motifsm35.enrichment(rgns_simp_vl, FigSubDir.TEST).is_file()


class TestPlotMotifs:
    def test_integrate_logos(self):
        assert PlotMotifs.integrate_logos().is_file()


class TestMotifsM30:
    def test_plot_ranked_tf(self):
        motifs = MotifsM30()
        figpath = motifs.plot_ranked_tf()
        assert figpath.is_file()

    def test_ranked_tf(self):
        motifs = MotifsM30()
        tfdf = motifs.ranked_tf()
        assert set(tfdf.columns) == set(["tf", "contrib_score"])

    def test_sorted_contrib(self):
        assert MotifsM30().sorted_contrib()[:4] == [71, 114, 74, 108]
