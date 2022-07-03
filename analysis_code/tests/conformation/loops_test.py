import subprocess

import pandas as pd
import pytest

from util.constants import CHRV_TOTAL_BP
from conformation.loops import (
    LoopReader,
    Loops,
    PlotLoops,
    LoopAnchors,
    COL_START,
    LoopInsides,
)
from chromosome.chromosome import Chromosome
from models.prediction import Prediction


@pytest.fixture
def ancrs_vl(chrm_vl_mean7: Chromosome):
    return LoopAnchors(chrm_vl_mean7, lim=250)


class TestLoopAnchors:
    def test_init(self, ancrs_vl: LoopAnchors):
        assert len(ancrs_vl) > 0

    def test_rm_close_ancrs(self, chrm_vl_mean7: Chromosome):
        ancrs = LoopAnchors(chrm_vl_mean7, lim=5)
        assert ancrs._rm_close_ancrs(pd.Series([3, 12, 14, 18, 27, 30])).tolist() == [
            3,
            12,
            18,
            27,
        ]


class TestLoopInsides:
    def test_loop_insides(self, ancrs_vl):
        insds_vl = LoopInsides(ancrs_vl)
        assert insds_vl.total_bp + ancrs_vl.total_bp == insds_vl.chrm.total_bp
        assert (
            ancrs_vl.mean_c0 * ancrs_vl.total_bp + insds_vl.mean_c0 * insds_vl.total_bp
        ) / (ancrs_vl.total_bp + insds_vl.total_bp) == pytest.approx(
            ancrs_vl.chrm.mean_c0, abs=1e-3
        )


class TestLoopReader:
    def test_read_loops(self):
        lpr = LoopReader(Chromosome("VL", None))
        df = lpr.read_loops()
        assert set(df.columns) == set(["start", "end", "res", "len"])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", lpr._loop_file])
        assert len(df) == int(s.split()[0]) - 2


class TestLoops:
    def test_len(self):
        loops = Loops(Chromosome("VL", None))
        assert len(loops) == len(loops._loop_df)

    def test_getitem(self):
        loops = Loops(Chromosome("VL", None))
        assert loops[10].tolist() == loops._loop_df.iloc[10].tolist()

    def test_add_mean_c0_val(self):
        """Assert mean C0 is: linker < full < nuc"""
        loops = Loops(Chromosome("VL"))
        mean_cols = loops.add_mean_c0()
        assert all(list(map(lambda col: col in loops._loop_df.columns, mean_cols)))
        assert loops._loop_df[mean_cols[2]].mean() < loops._loop_df[mean_cols[0]].mean()
        assert loops._loop_df[mean_cols[1]].mean() > loops._loop_df[mean_cols[0]].mean()

    def test_add_mean_c0_type_conserve(self):
        loops = Loops(Chromosome("VL"))
        loops.add_mean_c0()
        dtypes = loops._loop_df.dtypes
        assert dtypes["start"] == int
        assert dtypes["end"] == int
        assert dtypes["len"] == int

    def test_exclude_above_len(self):
        bf_loops = Loops(Chromosome("VL", None))
        bf_len = len(bf_loops._loop_df)
        bf_arr = bf_loops.covermask(bf_loops._loop_df)

        af_loops = Loops(Chromosome("VL", None), 100000)
        af_len = len(af_loops._loop_df)
        af_arr = af_loops.covermask(af_loops._loop_df)

        assert bf_len >= af_len
        assert bf_arr.sum() >= af_arr.sum()

    def test_covermask(self):
        loops = Loops(Chromosome("VL", None))
        chrm_arr = loops.covermask(loops._loop_df)
        assert chrm_arr.shape == (CHRV_TOTAL_BP,)
        perc = chrm_arr.sum() / CHRV_TOTAL_BP * 100
        assert 10 < perc < 90

    def test_slice(self, loops_vl: Loops):
        subloops = loops_vl[5:10]
        assert isinstance(subloops, Loops)
        assert len(subloops) == 5
        subloops._loop_df = None
        assert loops_vl._loop_df is not None


class TestPlotLoops:
    def test_line_plot_mean_c0(self):
        ploops = PlotLoops(Chromosome("VL"))
        paths = ploops.line_plot_mean_c0()
        for path in paths:
            assert path.is_file()

    def test_plot_mean_c0_across_loops(self):
        chr = Chromosome("VL", None)
        ploops = PlotLoops(chr)
        path = ploops.plot_mean_c0_across_loops(150)
        assert path.is_file()

    def test_plot_nuc_occ_across_loops(self):
        chr = Chromosome("II", Prediction(30))
        ploops = PlotLoops(chr)
        path = ploops.plot_mean_nuc_occupancy_across_loops()
        assert path.is_file()

    def test_plot_c0_around_anchor(self):
        ploops = PlotLoops(Chromosome("VL"))
        path = ploops.plot_c0_around_anchor(500)
        assert path.is_file()

    def test_plot_c0_around_indiv_ancr(self, chrm_vl_mean7: Chromosome):
        ploops = PlotLoops(chrm_vl_mean7)
        assert ploops.plot_c0_around_indiv_ancr(ploops._loops[5][COL_START]).is_file()

    def test_plot_scatter_mean_c0_vs_length(self):
        path = PlotLoops(Chromosome("VL")).plot_scatter_mean_c0_vs_length()
        assert path.is_file()
