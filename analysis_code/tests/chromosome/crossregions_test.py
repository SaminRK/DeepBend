from pathlib import Path

import pytest

from chromosome.chromosome import Chromosome
from chromosome.crossregions import (
    DistribC0DistPlot,
    DistribPlot,
    LineC0Plot,
    LinePlot,
    MCLineC0Plot,
    PlotPrmtrsBndrs,
    ScatterPlot,
    SegmentLineC0Plot,
)
from chromosome.genes import Promoters
from conformation.domains import BoundariesHE, BoundariesType


@pytest.fixture
def crplt_vl(chrm_vl_mean7: Chromosome):
    return DistribPlot(chrm_vl_mean7)


class TestDistribPlot:
    def test_prob_distrib_bndrs_nearest_ndr_distnc(self, crplt_vl: DistribPlot):
        assert crplt_vl.prob_distrib_bndrs_nearest_ndr_distnc().is_file()

    def test_distrib_cuml_bndrs_nearest_ndr_distnc(self, crplt_vl: DistribPlot):
        assert crplt_vl.distrib_cuml_bndrs_nearest_ndr_distnc().is_file()

    def test_num_prmtrs_bndrs_ndrs(self, crplt_vl: DistribPlot):
        assert crplt_vl.num_prmtrs_bndrs_ndrs(2, BoundariesType.FANC).is_file()

    def test_box_mean_c0_bndrs(self, crplt_vl: DistribPlot):
        assert crplt_vl.box_mean_c0_bndrs().is_file()

    def test_box_mean_c0_raw(self, crplt_vl: DistribPlot):
        assert crplt_vl.box_mean_c0(
            {
                "dls": [([2, 3, 6], "a"), ([7, 3, 8], "b")],
                "title": "test",
                "fname": "test.png",
            },
            "violin",
        )

    def test_prob_distrib_mean_c0_bndrs(self, crplt_vl: DistribPlot):
        assert crplt_vl.prob_distrib_mean_c0_bndrs().is_file()

    def test_prob_distrib_prmtrs_ndrs(self, crplt_vl: DistribPlot):
        assert crplt_vl.prob_distrib_prmtr_ndrs().is_file()

    def test_prob_distrib_len(self, crplt_vl: DistribPlot):
        assert crplt_vl.prob_distrib_linkers_len_in_prmtrs().is_file()


class TestDistribC0DistPlot:
    def test_bndrs(self):
        assert DistribC0DistPlot(Chromosome("VL")).bndrs().is_file()


@pytest.fixture
def lnplt_vl(chrm_vl_mean7: Chromosome):
    return LineC0Plot(chrm_vl_mean7)


class TestScatterPlot:
    def test_scatter_c0(self, chrm_vl_mean7: Chromosome):
        assert ScatterPlot(chrm_vl_mean7).scatter_c0().is_file()

    def test_scatter_kmer(self, chrm_vl_mean7: Chromosome):
        assert ScatterPlot(chrm_vl_mean7).scatter_kmer("TTTT").is_file()


class TestLineC0Plot:
    def test_mean_bndrs(self, lnplt_vl: LineC0Plot):
        lnplt_vl.line_c0_mean_bndrs(200).is_file()

    def test_line_c0_bndry_indiv_toppings(
        self, lnplt_vl: LineC0Plot, bndrs_hirs_vl: BoundariesHE
    ):
        assert lnplt_vl._line_c0_bndry_indiv_toppings(
            bndrs_hirs_vl.bnds_in_prms()[5], str(bndrs_hirs_vl), "prmtr"
        ).is_file()

    def test_line_c0_prmtr_indiv_toppings(
        self, lnplt_vl: LineC0Plot, prmtrs_vl: Promoters
    ):
        assert lnplt_vl._line_c0_prmtr_indiv_toppings(prmtrs_vl[10]).is_file()

    def test_line_c0_toppings(self, lnplt_vl: LineC0Plot):
        assert lnplt_vl.line_c0_toppings(342000, 343000).is_file()

    def test_text_pos_calc(self, lnplt_vl: LineC0Plot):
        assert list(lnplt_vl._text_pos_calc(5, 24, 0.04)) == [
            (5, 0.04),
            (9, 0.02),
            (13, -0.02),
            (17, -0.04),
            (21, 0.04),
        ]

class TestMCLineC0Plot:
    def test_line_c0(self):
        assert MCLineC0Plot().line_c0_mean_bndrs().is_file()
class TestSegmentLineC0Plot:
    def test_sl_lnkrs(self, chrm_vl_mcvr):
        sl = SegmentLineC0Plot(chrm_vl_mcvr)
        assert sl.sl_lnkrs().is_file()


class TestLinePlot:
    def test_dinc_content_bndrs(self, chrm_vl_mean7: Chromosome):
        assert LinePlot(chrm_vl_mean7).dinc_mean_bndrs().is_file()

    def test_helsep_mean_bndrs(self, chrm_vl_mean7: Chromosome):
        assert LinePlot(chrm_vl_mean7).helsep_mean_bndrs().is_file()


class TestPlotPrmtrsBndrs:
    def test_dinc_explain_scatter(self):
        pltpb = PlotPrmtrsBndrs()
        assert pltpb.dinc_explain_scatter().is_file()

    def test_dinc_explain_box(self):
        pltpb = PlotPrmtrsBndrs()
        assert pltpb.dinc_explain_box().is_file()

    def test_both_motif_contrib_single(self):
        pltpb = PlotPrmtrsBndrs()
        if not Path(pltpb._contrib_file(pltpb.WB_DIR, 0, "png")).is_file():
            return

        assert pltpb._both_motif_contrib_single(pltpb.BOTH_DIR, 0).is_file()
