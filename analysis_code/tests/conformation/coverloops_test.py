from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformation.coverloops import (
    CoverLoops,
    MCCoverLoops,
    NonCoverLoops,
    PlotCoverLoops,
    MultiChrmLoopsCoverCollector,
    PlotMCCoverLoops,
)
from conformation.loops import COL_MEAN_C0_FULL, Loops, COL_START, COL_END, MCLoops
from chromosome.chromosome import Chromosome, MultiChrm
from util.constants import ONE_INDEX_START


@pytest.fixture
def cloops_vl(loops_vl):
    return CoverLoops(loops_vl)


class TestCoverLoops:
    def test_coverloops(self, cloops_vl: CoverLoops):
        cloops_vl.covermask = np.array(
            [False, False, True, False, True, True, False, True, False]
        )
        df = cloops_vl._coverloops()
        assert df[COL_START].tolist() == [3, 5, 8]
        assert df[COL_END].tolist() == [3, 6, 8]

    def test_iter(self, cloops_vl):
        cloops = [cl for cl in cloops_vl]
        assert len(cloops) > 0
        assert getattr(cloops[0], COL_START) > ONE_INDEX_START

    def test_access(self, cloops_vl):
        assert len(cloops_vl) == len(cloops_vl[COL_MEAN_C0_FULL])


class TestNonCoverLoops:
    def test_noncoverloops(self, loops_vl, cloops_vl):
        ncloops = NonCoverLoops(loops_vl)
        assert len(ncloops) == len(cloops_vl) + 1


class TestPlotCoverLoops:
    def test_plot_histogram_c0(self, chrm_vl):
        ploops = PlotCoverLoops(chrm_vl)
        figpath = ploops.plot_histogram_c0()
        assert figpath.is_file()


@pytest.fixture
def mchrm_vl_i():
    return MultiChrm(("VL", "I"))


class TestMCCoverLoops:
    def test_creation(self, mchrm_vl_i, cloops_vl, chrm_i):
        mccl = MCCoverLoops(MCLoops(mchrm_vl_i))
        cli = CoverLoops(Loops(chrm_i))
        assert len(mccl) == len(cloops_vl) + len(cli)


class TestPlotMCCoverLoops:
    def test_box_plot(self, mchrm_vl_i):
        assert PlotMCCoverLoops(mchrm_vl_i).box_plot_c0().is_file()

    def test_histogram(self, mchrm_vl_i):
        figpath = PlotMCCoverLoops(mchrm_vl_i).plot_histogram_c0()
        assert figpath.is_file()


class TestMultiChrmLoopsCoverCollector:
    @pytest.fixture
    def mclccoll_vl(self):
        return MultiChrmLoopsCoverCollector(("VL",), 1000000)

    def test_get_cover_stat(self, mclccoll_vl):
        colt_df, path_str = mclccoll_vl.get_cover_stat()
        assert Path(path_str).is_file()
        assert pd.read_csv(path_str, sep="\t").columns.tolist() == [
            "ChrID",
            "loop_nuc",
            "loop_linker",
            "non_loop_nuc",
            "non_loop_linker",
        ]

    def test_plot_cover_stat(self, mclccoll_vl):
        path_str = mclccoll_vl.plot_bar_cover_stat()
        assert Path(path_str).is_file()
