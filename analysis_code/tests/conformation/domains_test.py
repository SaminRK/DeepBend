from telnetlib import DO
import pytest
import pandas as pd

from conformation.domains import (
    BndParm,
    Boundaries,
    BoundariesFN,
    BoundariesHE,
    PlotBoundariesHE,
    MCBoundariesHEAggregator,
    MCBoundariesHECollector,
    BoundariesF,
    DomainsF,
    BoundariesFactory,
    BoundariesType,
    BndFParm,
    BndSel,
)
from chromosome.regions import MIDDLE, START, END, LEN, Regions
from chromosome.chromosome import Chromosome
from models.prediction import Prediction


class TestBoundaries:
    def test_nearest_rgns(self, chrm_vl_mcvr: Chromosome, rgns_simp_vl_2: Regions):
        nr = Boundaries(
            chrm_vl_mcvr, pd.DataFrame({START: [3, 5, 10], END: [7, 13, 14]})
        ).nearest_rgns(rgns_simp_vl_2)
        assert nr[START].tolist() == [4, 9]
        assert nr[END].tolist() == [6, 11]


@pytest.fixture
def bndrs_vl(chrm_vl_mean7):
    return BoundariesHE(chrm_vl_mean7, res=500, lim=250)


class TestBoundariesHE:
    def test_hires_strong(self, chrm_vl_mean7):
        bndrs = BoundariesHE(chrm_vl_mean7, **BndParm.HIRS_WD_50)
        assert len(bndrs) > 0

    def test_nearest_locs_distnc(self, bndrs_vl: BoundariesHE):
        bndrs_vl._regions = pd.DataFrame({MIDDLE: [18, 40, 4]})
        assert bndrs_vl.nearest_locs_distnc([30, 9, 26]).tolist() == [8, -10, 5]

    def test_read_boundaries_of(self, bndrs_vl: BoundariesHE):
        assert len(bndrs_vl) == 57

    def test_mean_c0(self, bndrs_vl: BoundariesHE):
        mc0 = bndrs_vl.mean_c0
        assert mc0 == pytest.approx(-0.180, abs=1e-2)

    def test_extended(self, bndrs_vl):
        ebndrs = bndrs_vl.extended(50)
        assert (bndrs_vl[START] - ebndrs[START]).tolist() == [50] * len(ebndrs)
        assert (ebndrs[END] - bndrs_vl[END]).tolist() == [50] * len(ebndrs)
        assert (ebndrs[LEN] - bndrs_vl[LEN]).tolist() == [100] * len(ebndrs)

    def test_prmtr_non_prmtr_bndrs(self, bndrs_vl: BoundariesHE):
        prmtr_bndrs = bndrs_vl.prmtr_bndrs()
        non_prmtr_bndrs = bndrs_vl.non_prmtr_bndrs()
        assert -0.5 < prmtr_bndrs.mean_c0 < 0
        assert -0.5 < non_prmtr_bndrs.mean_c0 < 0
        assert prmtr_bndrs.mean_c0 > non_prmtr_bndrs.mean_c0

        assert bndrs_vl.mean_c0 == pytest.approx(
            (
                prmtr_bndrs.mean_c0 * len(prmtr_bndrs)
                + non_prmtr_bndrs.mean_c0 * len(non_prmtr_bndrs)
            )
            / len(bndrs_vl),
            rel=1e-3,
        )


@pytest.fixture
def bndrsf_vl(chrm_vl_mean7):
    return BoundariesFactory(chrm_vl_mean7).get_bndrs(
        BndSel(BoundariesType.FANC, BndFParm.SHR_25)
    )


class TestBoundariesF:
    def test_init(self, chrm_vl_mean7):
        bndrs = BoundariesF(chrm_vl_mean7, top_perc=0.25)
        assert len(bndrs) == 45

    def test_extended(self, bndrsf_vl):
        ebndrsf = bndrsf_vl.extended(50)
        assert (bndrsf_vl[START] - ebndrsf[START]).tolist() == [50] * len(ebndrsf)
        assert (ebndrsf[END] - bndrsf_vl[END]).tolist() == [50] * len(ebndrsf)
        assert (ebndrsf[LEN] - bndrsf_vl[LEN]).tolist() == [100] * len(ebndrsf)


class TestDomainsF:
    def test_domains(self, bndrsf_vl: BoundariesF):
        domf_vl = DomainsF(bndrsf_vl)
        assert domf_vl.total_bp + bndrsf_vl.total_bp == domf_vl.chrm.total_bp
        assert (
            bndrsf_vl.mean_c0 * bndrsf_vl.total_bp + domf_vl.mean_c0 * domf_vl.total_bp
        ) / (bndrsf_vl.total_bp + domf_vl.total_bp) == pytest.approx(
            bndrsf_vl.chrm.mean_c0, abs=1e-3
        )

    def test_sections(self, bndrsf_vl: BoundariesF):
        dmnsf = DomainsF(bndrsf_vl)
        assert len(dmnsf.sections(200)) > 0


class TestBoundariesFN:
    def test_init(self, chrm_vl_mcvr: Chromosome):
        bndrsf = BoundariesF(chrm_vl_mcvr, **BndFParm.SHR_50)
        bndrsfn = BoundariesFN(chrm_vl_mcvr, **BndFParm.SHR_50)
        assert len(bndrsfn) == len(bndrsf)


@pytest.mark.skip(reason="Updating domains")
class TestBoundariesDomainsHEQuery:
    def test_num_greater_than_dmns(self):
        bndrs = BoundariesHE(Chromosome("VI", Prediction(30)))
        bndrs_gt = bndrs.num_bndry_mean_c0_greater_than_dmn()
        prmtr_bndrs_gt = bndrs.num_prmtr_bndry_mean_c0_greater_than_dmn()
        non_prmtr_bndrs_gt = bndrs.num_non_prmtr_bndry_mean_c0_greater_than_dmns()
        assert bndrs_gt == prmtr_bndrs_gt + non_prmtr_bndrs_gt


@pytest.fixture
def plotbndrs_vl(chrm_vl):
    return PlotBoundariesHE(chrm_vl)


class TestPlotBoundariesHE:
    def test_plot_scatter_mean_c0_each_bndry(self, plotbndrs_vl: PlotBoundariesHE):
        figpath = plotbndrs_vl.scatter_mean_c0_at_indiv()
        assert figpath.is_file()

    def test_line_c0_around_indiv(self, plotbndrs_vl: PlotBoundariesHE):
        assert plotbndrs_vl._line_c0_around_indiv(plotbndrs_vl._bndrs[7], "").is_file()


@pytest.mark.skip(reason="Updating domains")
class TestMCBoundariesHECollector:
    def test_add_dmns_mean(self):
        coll = MCBoundariesHECollector(Prediction(30), ("VI", "VII"))
        c0_dmns_col = coll._add_dmns_mean()
        assert c0_dmns_col in coll._coll_df.columns
        assert all(coll._coll_df[c0_dmns_col] > -0.4)
        assert all(coll._coll_df[c0_dmns_col] < -0.1)

    def test_add_num_bndrs_gt_dmns(self):
        coll = MCBoundariesHECollector(Prediction(30), ("VI", "VII"))
        num_bndrs_gt_dmns_col = coll._add_num_bndrs_gt_dmns()
        assert num_bndrs_gt_dmns_col in coll._coll_df.columns
        assert all(coll._coll_df[num_bndrs_gt_dmns_col] > 10)
        assert all(coll._coll_df[num_bndrs_gt_dmns_col] < 200)

    def test_save_stat(self):
        # TODO *: Use default prediction 30
        coll = MCBoundariesHECollector(Prediction(30), ("VII", "X"))
        path = coll.save_stat([0, 3, 4])
        assert path.is_file()

    def test_plot_scatter_mean_c0(self):
        mcbndrs = MCBoundariesHECollector(Prediction(), ("VII", "XII", "XIII"))
        path = mcbndrs.plot_scatter_mean_c0()
        assert path.is_file()

    def test_plot_bar_perc_in_prmtrs(self):
        mcbndrs = MCBoundariesHECollector(Prediction(30), ("VII", "XII", "XIII"))
        path = mcbndrs.plot_bar_perc_in_prmtrs()
        assert path.is_file()

    def test_mean_dmn_len(self):
        """
        Test
            * If two mean boundaries are withing +-10%.
            * mean_dmn is within 7000 - 12000bp
        """
        mcbndrs_xi_ii = MCBoundariesHECollector(Prediction(30), ("XI", "II"))
        mcbndrs_vii = MCBoundariesHECollector(Prediction(30), ("VII",))
        xi_ii_mean_dmn = mcbndrs_xi_ii.mean_dmn_len()
        vii_mean_dmn = mcbndrs_vii.mean_dmn_len()
        assert xi_ii_mean_dmn > vii_mean_dmn * 0.9
        assert xi_ii_mean_dmn < vii_mean_dmn * 1.1
        assert xi_ii_mean_dmn > 7000
        assert xi_ii_mean_dmn < 12000


@pytest.mark.skip(reason="Updating domains")
class TestMCBoundariesHEAggregator:
    def test_save_stat(self):
        aggr = MCBoundariesHEAggregator(
            MCBoundariesHECollector(Prediction(30), ("IX", "XIV"))
        )
        path = aggr.save_stat()
        assert path.is_file()
