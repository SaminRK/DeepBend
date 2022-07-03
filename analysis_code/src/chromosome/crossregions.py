from __future__ import annotations
import random
import math
from pathlib import Path
from enum import Enum, auto
from collections import namedtuple
from typing import Any, Callable, Iterator, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, ConnectionPatch
import matplotlib.ticker as plticker
import pandas as pd
import numpy as np
from cairosvg import svg2png
from scipy.ndimage import gaussian_filter1d
from skimage.transform import resize
from nptyping import NDArray
from statsmodels.stats.weightstats import ztest

from chromosome.chromosome import C0Spread, ChrmCalc, PlotChrm, ChrmOperator, Chromosome
from chromosome.genes import Genes, Promoters, STRAND
from chromosome.regions import END, MIDDLE, START, LEN, MEAN_C0, Regions
from chromosome.nucleosomes import Linkers, Nucleosomes
from models.prediction import Prediction
from motif.motifs import MOTIF_NO, N_MOTIFS, P_VAL, ZTEST_VAL, MotifsM30, MotifsM35
from conformation.domains import (
    BndParmT,
    Boundaries,
    BoundariesF,
    BoundariesFN,
    BoundariesHE,
    SCORE,
    BndParm,
    BoundariesType,
    BoundariesFactory,
    BndFParm,
    BndSel,
    DomainsF,
    DomainsFN,
    DomainsHE,
)
from conformation.loops import LoopAnchors, LoopInsides
from feature_model.helsep import DincUtil, HelSep, SEQ_COL
from util.util import Attr, PathObtain, PlotUtil, FileSave, rev_comp
from util.custom_types import PosOneIdx, KMerSeq
from util.kmer import KMer
from util.constants import (
    SEQ_LEN,
    FigSubDir,
    ONE_INDEX_START,
    GDataSubDir,
    YeastChrNumList,
)


class SubRegions:
    def __init__(self, chrm: Chromosome) -> None:
        self.chrm = chrm
        self._prmtrs = None
        self._bndrs = None
        self.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
        self.min_ndr_len = 40

    @property
    def bndrs(self) -> Boundaries:
        def _bndrs():
            return BoundariesFactory(self.chrm).get_bndrs(self.bsel)

        return Attr.calc_attr(self, "_bndrs", _bndrs)

    @property
    def bndsf(self) -> BoundariesF:
        def _bndsf():
            return BoundariesF(self.chrm, **BndFParm.SHR_50)

        return Attr.calc_attr(self, "_bndsf", _bndsf)

    @property
    def bndsfn(self) -> BoundariesF:
        def _bndsfn():
            return BoundariesFN(self.chrm, **BndFParm.SHR_50_LNK_0)

        return Attr.calc_attr(self, "_bndsfn", _bndsfn)

    @property
    def dmns(self) -> Promoters:
        def _dmns():
            if isinstance(self.bndrs, BoundariesHE):
                D = DomainsHE
            elif isinstance(self.bndrs, BoundariesF):
                D = DomainsF
            elif isinstance(self.bndrs, BoundariesFN):
                D = DomainsFN

            return D(self.bndrs)

        return Attr.calc_attr(self, "_dmns", _dmns)

    @property
    def dmnsf(self) -> DomainsF:
        def _dmnsf():
            return DomainsF(self.bndsf)

        return Attr.calc_attr(self, "_dmnsf", _dmnsf)

    @property
    def dmnsfn(self) -> DomainsFN:
        def _dmnsfn():
            return DomainsFN(self.bndsfn)

        return Attr.calc_attr(self, "_dmnsfn", _dmnsfn)

    @property
    def prmtrs(self) -> Promoters:
        def _prmtrs():
            return Promoters(self.chrm, ustr_tss=499, dstr_tss=100)

        return Attr.calc_attr(self, "_prmtrs", _prmtrs)

    @property
    def genes(self) -> Genes:
        def _genes():
            return Genes(self.chrm)

        return Attr.calc_attr(self, "_genes", _genes)

    @property
    def nucs(self) -> Nucleosomes:
        def _nucs():
            return Nucleosomes(self.chrm)

        return Attr.calc_attr(self, "_nucs", _nucs)

    @property
    def lnkrs(self) -> Linkers:
        def _lnkrs():
            return Linkers(self.chrm)

        return Attr.calc_attr(self, "_lnkrs", _lnkrs)

    @property
    def ndrs(self) -> Linkers:
        def _ndrs():
            return self.lnkrs.ndrs(self.min_ndr_len)

        return Attr.calc_attr(self, "_ndrs", _ndrs)

    @property
    def lpancrs(self):
        return LoopAnchors(self.chrm, lim=250)

    @property
    def lpinsds(self):
        return LoopInsides(self.lpancrs)

    @property
    def bsel(self) -> BndSel:
        return self._bsel

    @bsel.setter
    def bsel(self, _bsel: BndSel):
        self._bsel = _bsel
        self._bndrs = None

    def prmtrs_with_bndrs(self) -> Promoters:
        return self.prmtrs.with_loc(self.bndrs[MIDDLE], True)

    def prmtrs_wo_bndrs(self) -> Promoters:
        return self.prmtrs.with_loc(self._bndrs[MIDDLE], False)

    def prmtrs_in_bnds(self) -> Promoters:
        return self.prmtrs.mid_contained_in(self.bndrs)

    def prmtrs_in_dmns(self) -> Promoters:
        return self.prmtrs.mid_contained_in(self.dmns)

    def bnds_in_prms(self):
        # TODO: Update def. in +- 100bp of promoters
        return self.bndrs.mid_contained_in(self.prmtrs)

    def bnds_in_nprms(self):
        return self.bndrs - self.bnds_in_prms()

    def bndry_nucs(self) -> Nucleosomes:
        return self.nucs.overlaps_with_rgns(self.bndrs, 50)

    def non_bndry_nucs(self) -> Nucleosomes:
        return self.nucs - self.bndry_nucs()

    def lnks_in_bnds(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.bndrs)

    def lnks_in_bndsf(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.bndsf)

    def lnks_in_dmns(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.dmns)

    def lnks_in_dmnsf(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.dmnsf)

    def nucs_in_bnds(self) -> Nucleosomes:
        n = self.nucs.mid_contained_in(self.bndrs)
        n.str = f"{n}_{self.bndrs}"
        return n

    def nucs_in_bndsf(self) -> Nucleosomes:
        n = self.nucs.mid_contained_in(self.bndsf)
        n.str = f"{n}_{self.bndsf}"
        return n

    def nucs_in_dmns(self) -> Nucleosomes:
        n = self.nucs.mid_contained_in(self.dmns)
        n.str = f"{n}_{self.dmns}"
        return n

    def nucs_in_dmnsf(self) -> Nucleosomes:
        n = self.nucs.mid_contained_in(self.dmnsf)
        n.str = f"{n}_{self.dmnsf}"
        return n

    def bndry_ndrs(self) -> Linkers:
        return self.ndrs.overlaps_with_rgns(self.bndrs, self.min_ndr_len)

    def non_bndry_ndrs(self) -> Linkers:
        return self.ndrs - self.bndry_ndrs()


sr_vl = SubRegions(Chromosome("VL", prediction=None, spread_str=C0Spread.mcvr))


class Distrib(Enum):
    BNDRS = auto()
    BNDRS_E_100 = auto()
    BNDRS_E_N50 = auto()
    NUCS = auto()
    NUCS_B = auto()
    NUCS_NB = auto()
    LNKRS = auto()
    NDRS = auto()
    NDRS_B = auto()
    NDRS_NB = auto()
    PRMTRS = auto()
    GENES = auto()
    BNDRS_P = auto()
    BNDRS_NP = auto()
    PRMTRS_B = auto()
    PRMTRS_NB = auto()
    LPANCRS = auto()
    LPINSDS = auto()


class LabeledMC0Distribs:
    def __init__(self, sr: SubRegions):
        self._sr = sr

    def dl(self, ds: list[Distrib]) -> list[tuple[np.ndarray, str]]:
        def _dl(d: Distrib):
            if d == Distrib.BNDRS:
                return self._sr.bndrs[MEAN_C0], "bndrs l 100"
            if d == Distrib.BNDRS_E_100:
                return self._sr.bndrs.extended(100)[MEAN_C0], "bndrs l 200"
            if d == Distrib.BNDRS_E_N50:
                return self._sr.bndrs.extended(-50)[MEAN_C0], "bndrs l 50"
            if d == Distrib.NUCS:
                return self._sr.nucs[MEAN_C0], "Nucleosomes"
            if d == Distrib.NUCS_B:
                return self._sr.bndry_nucs()[MEAN_C0], "Nucleosomes\nin boundaries"
            if d == Distrib.NUCS_NB:
                return self._sr.non_bndry_nucs()[MEAN_C0], "Nucleosomes\nin domains"
            if d == Distrib.LNKRS:
                return self._sr.lnkrs[MEAN_C0], "lnkrs"
            if d == Distrib.NDRS:
                return self._sr.ndrs[MEAN_C0], "NDRs"
            if d == Distrib.NDRS_B:
                return self._sr.bndry_ndrs()[MEAN_C0], "NDRs in boundaries"
            if d == Distrib.NDRS_NB:
                return self._sr.non_bndry_ndrs()[MEAN_C0], "NDRs in domains"
            if d == Distrib.PRMTRS:
                return self._sr.prmtrs[MEAN_C0], "prmtrs"
            if d == Distrib.GENES:
                return self._sr.genes[MEAN_C0], "genes"
            if d == Distrib.BNDRS_P:
                return self._sr.bnds_in_prms()[MEAN_C0], "bndrs p"
            if d == Distrib.BNDRS_NP:
                return self._sr.bnds_in_nprms()[MEAN_C0], "bndrs np"
            if d == Distrib.PRMTRS_B:
                return self._sr.prmtrs_with_bndrs()[MEAN_C0], "prmtrs b"
            if d == Distrib.PRMTRS_NB:
                return self._sr.prmtrs_wo_bndrs()[MEAN_C0], "prmtrs nb"
            if d == Distrib.LPANCRS:
                return self._sr.lpancrs[MEAN_C0], "lp ancrs"
            if d == Distrib.LPINSDS:
                return self._sr.lpinsds[MEAN_C0], "lp insds"

        return list(map(_dl, ds))


class DistribPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self._sr = SubRegions(chrm)

    def box_mean_c0_bndrs(self) -> Path:
        typ = "box"
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        sr.bsel = bsel_fanc
        ld = LabeledMC0Distribs(sr)
        grp_bndrs_nucs = {
            "dls": ld.dl(
                [
                    # Distrib.BNDRS,
                    # Distrib.BNDRS_E_100,
                    # Distrib.BNDRS_E_N50,
                    # Distrib.NUCS,
                    # Distrib.NUCS_B,
                    # Distrib.NUCS_NB,
                    # Distrib.LNKRS,
                    Distrib.NDRS,
                    Distrib.NDRS_B,
                    Distrib.NDRS_NB,
                ]
            ),
            "title": "",  # "Mean C0 distribution of boundaries, nucleosomes and NDRS",
            "fname": f"bndrs_nucs_{sr.bndrs}_{typ}_chrm_{sr.chrm}.png",
        }
        grp_bndrs_prmtrs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.PRMTRS,
                    Distrib.GENES,
                    Distrib.BNDRS_P,
                    Distrib.BNDRS_NP,
                    Distrib.PRMTRS_B,
                    Distrib.PRMTRS_NB,
                ]
            ),
            "title": "Mean C0 distrib of comb of prmtrs and bndrs",
            "fname": f"bndrs_prmtrs_{sr.bndrs}_{sr.prmtrs}_{self._chrm.id}.png",
        }
        return self.box_mean_c0(grp_bndrs_nucs, typ)

    @classmethod
    def box_mean_c0(cls, grp: dict, typ: Literal["box"] | Literal["violin"] = "box"):
        limy = False
        PlotUtil.font_size(14)
        PlotUtil.show_grid()
        distribs = [d for d, _ in grp["dls"]]
        labels = [l for _, l in grp["dls"]]

        fig, ax = plt.subplots()
        if typ == "box":
            plt.boxplot(distribs, showfliers=True, widths=0.5)
        elif typ == "violin":
            ax.violinplot(distribs, showmedians=True, showextrema=True)

        if limy:
            plt.ylim(-0.5, 0.1)
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, wrap=True)
        plt.ylabel("Mean C0")
        plt.title(grp["title"])
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/c0_box/{grp['fname']}", sizew=7, sizeh=6
        )

    def box_bnd_dmn_lnklen(self):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
        bl = sr.lnkrs.mid_contained_in(sr.bndrs)
        bqs = sr.bndrs.quartiles()
        blqs = [sr.lnkrs.mid_contained_in(bq) for bq in bqs]
        dl = sr.lnkrs.mid_contained_in(sr.dmns)
        assert sum([len(blq) for blq in blqs]) == len(bl)
        assert len(bl) + len(dl) == len(sr.lnkrs)
        PlotUtil.font_size(20)
        plt.boxplot([blq[LEN] for blq in blqs] + [dl[LEN]], showfliers=False)
        # plt.boxplot([bl[LEN], dl[LEN]], showfliers=True, widths=0.5)
        plt.xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Domains"])
        plt.ylabel("Linker length (bp)")
        plt.tight_layout()
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{sr.bndrs}/lnklen_box_bnd_dmn.png"
        )

    def prob_distrib_mean_c0_bndrs(self):
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_25)
        sr.bsel = bsel_hexp
        ld = LabeledMC0Distribs(sr)
        grp_bndrs_nucs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.BNDRS_E_N50,
                    Distrib.LNKRS,
                    Distrib.NDRS,
                    Distrib.NDRS_B,
                    Distrib.NDRS_NB,
                ]
            ),
            "title": "Prob distrib of mean C0 distrib of bndrs and nucs",
            "fname": f"bndrs_nucs_{sr.bndrs}_{self._chrm.id}.png",
        }
        grp_bndrs_prmtrs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.PRMTRS,
                    Distrib.GENES,
                    Distrib.BNDRS_P,
                    Distrib.BNDRS_NP,
                    Distrib.PRMTRS_B,
                    Distrib.PRMTRS_NB,
                ]
            ),
            "title": "Prob distrib of mean C0 distrib of comb of prmtrs and bndrs",
            "fname": f"bndrs_prmtrs_{sr.bndrs}_{sr.prmtrs}_{self._chrm.id}.png",
        }
        grp = grp_bndrs_nucs

        PlotUtil.clearfig()
        PlotUtil.show_grid()

        for d, l in grp["dls"]:
            PlotUtil.prob_distrib(d, l)

        plt.legend()
        plt.xlabel("Mean c0")
        plt.ylabel("Probability")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/prob_distrib/{grp['fname']}"
        )

    def distrib_cuml_bndrs_nearest_tss_distnc(self) -> Path:
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)
        self._distrib_cuml_nearest_tss_distnc(bndrs)
        plt.title(
            f"Cumulative perc. of distance from boundary res={bndrs.res} bp "
            f"middle to nearest TSS"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_tss_distrib_cuml_res_{bndrs.res}_"
            f"perc_{bndrs.score_perc}"
            f"_{self._chrm.number}.png"
        )

    def distrib_cuml_random_locs_nearest_tss_distnc(self) -> Path:
        rndlocs = [
            random.randint(ONE_INDEX_START, self._chrm.total_bp) for _ in range(1000)
        ]
        locs = BoundariesHE(
            self._chrm, regions=pd.DataFrame({START: rndlocs, END: rndlocs})
        )
        self._distrib_cuml_nearest_tss_distnc(locs)
        plt.title(f"Cumulative perc. of distance from random pos to nearest TSS")
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_tss_random_pos_distrib_cuml_{self._chrm.number}.png"
        )

    def _distrib_cuml_nearest_tss_distnc(self, bndrs: BoundariesHE):
        genes = Genes(self._chrm)
        distns = bndrs.nearest_locs_distnc(
            pd.concat([genes.frwrd_genes()[START], genes.rvrs_genes()[END]])
        )

        PlotUtil.distrib_cuml(distns)
        plt.xlim(-1000, 1000)
        PlotUtil.show_grid()
        plt.xlabel("Distance")
        plt.ylabel("Percentage")

    def distrib_cuml_bndrs_nearest_ndr_distnc(
        self, min_lnker_len: list[int] = [80, 60, 40, 30]
    ) -> Path:
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)
        self._distrib_cuml_nearest_ndr_distnc(bndrs, min_lnker_len)
        plt.title(
            f"Cumulative perc. of distance from boundary res={bndrs.res} bp "
            f"middle to nearest NDR >= x bp"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_ndr_distrib_cuml_res_{bndrs.res}_"
            f"perc_{bndrs.score_perc}_{'_'.join(str(i) for i in min_lnker_len)}"
            f"_{self._chrm.number}.png"
        )

    def distrib_cuml_random_locs_nearest_ndr_distnc(
        self, min_lnker_len: list[int] = [80, 60, 40, 30]
    ) -> Path:
        rndlocs = [
            random.randint(ONE_INDEX_START, self._chrm.total_bp) for _ in range(1000)
        ]
        locs = BoundariesHE(
            self._chrm, regions=pd.DataFrame({START: rndlocs, END: rndlocs})
        )
        self._distrib_cuml_nearest_ndr_distnc(locs, min_lnker_len)
        plt.title(
            f"Cumulative perc. of distance from random pos" f" to nearest NDR >= x bp"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_ndr_random_pos_distrib_cuml"
            f"_{'_'.join(str(i) for i in min_lnker_len)}"
            f"_{self._chrm.number}.png"
        )

    def _distrib_cuml_nearest_ndr_distnc(
        self, bndrs: BoundariesHE, min_lnker_len: list[int]
    ):
        lnkrs = Linkers(self._chrm)
        for llen in min_lnker_len:
            distns = bndrs.nearest_locs_distnc(lnkrs.ndrs(llen)[MIDDLE])
            PlotUtil.distrib_cuml(distns, label=str(llen))

        plt.legend()
        plt.xlim(-1000, 1000)
        PlotUtil.show_grid()
        plt.xlabel("Distance")
        plt.ylabel("Percentage")

    def prob_distrib_bndrs_nearest_ndr_distnc(self, min_lnker_len: int) -> Path:
        allchrm = True
        llen = min_lnker_len
        if allchrm:
            dists = np.empty((0,))
            pred = Prediction(35)
            for c in YeastChrNumList:
                print(c)
                chrm = Chromosome(c, prediction=pred, spread_str=C0Spread.mcvr)
                self._sr = SubRegions(chrm)
                self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
                dists = np.append(
                    dists,
                    self._sr.bndrs.nearest_locs_distnc(
                        self._sr.lnkrs.ndrs(llen)[MIDDLE]
                    ),
                )
        else:
            self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
            dists = self._sr.bndrs.nearest_locs_distnc(
                self._sr.lnkrs.ndrs(llen)[MIDDLE]
            )

        in_bnd = round(
            np.sum((-self._sr.bndrs.lim < dists) & (dists <= self._sr.bndrs.lim))
            / len(dists)
            * 100,
            1,
        )
        lft_bnd = round(np.sum(dists <= -self._sr.bndrs.lim) / len(dists) * 100, 1)
        rgt_bnd = round(np.sum(self._sr.bndrs.lim < dists) / len(dists) * 100, 1)
        PlotUtil.font_size(20)
        PlotUtil.prob_distrib(dists, label=str(llen))
        PlotUtil.vertline(-self._sr.bndrs.lim + 1, "k", linewidth=2)
        PlotUtil.vertline(self._sr.bndrs.lim, "k", linewidth=2)
        ax = plt.gca()
        ax.set_xlim([-250, 250])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        plt.text(0, sum(ylim) / 2, f"{in_bnd}%")
        plt.text((xlim[0] + -self._sr.bndrs.lim) / 2, sum(ylim) / 2, f"{lft_bnd}%")
        plt.text((xlim[1] + self._sr.bndrs.lim) / 2, sum(ylim) / 2, f"{rgt_bnd}%")

        # PlotUtil.show_grid()
        plt.xlabel("Distance from boundary middle (bp)")
        plt.ylabel("Density")
        # plt.title(
        #     f"Prob distrib of distance from boundary res={self._sr.bndrs.res} bp "
        #     f"middle to nearest NDR >= x bp"
        # )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/distnc_ndr_prob_distrib_{self._sr.bndrs}_"
            f"{f'all{str(self._chrm)[:-4]}' if allchrm else self._chrm}.png"
        )

    def prob_distrib_bndrs_nearest_tss_dist(self):
        hist = True
        self._sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
        dists = self._sr.bndrs.nearest_tss_distnc(self._sr.genes)
        PlotUtil.font_size(20)
        if hist:
            bins = 20
            plt.hist(dists, bins=bins)
        else:
            PlotUtil.prob_distrib(dists)
        plt.xlabel("Distance from TSS (bp)")
        plt.ylabel("Density")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.GENES}/{self._chrm}_{self._sr.genes}/distnc_bndr_prob_distrib_{self._sr.bndrs}"
            f"{'_hist' if hist else '_density'}.png"
        )

    def num_prmtrs_bndrs_ndrs(self, frml: int, btype: BoundariesType) -> Path:
        min_lnkr_len = 40

        def _includes(*rgnss):
            if frml == 1:
                return Regions.with_rgn(*rgnss)
            elif frml == 2:
                return Regions.overlaps_with_rgns(*rgnss, min_ovbp=min_lnkr_len)

        ndrs = Linkers(self._chrm).ndrs(min_lnkr_len)
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = _includes(prmtrs, ndrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr

        if btype == BoundariesType.HEXP:
            bparm = BndParm.HIRS_SHR_50
        elif btype == BoundariesType.FANC:
            bparm = BndFParm.SHR_25

        bndrs = BoundariesFactory(self._chrm).get_bndrs(BndSel(btype, bparm))
        bndrs_with_ndr = _includes(bndrs, ndrs)
        bndrs_wo_ndr = bndrs - bndrs_with_ndr
        ndrs_in_prmtrs = _includes(ndrs, prmtrs)
        ndrs_out_prmtrs = ndrs - ndrs_in_prmtrs
        ndrs_in_bndrs = _includes(ndrs, bndrs)
        ndrs_out_bndrs = ndrs - ndrs_in_bndrs
        PlotUtil.clearfig()
        PlotUtil.show_grid("major")
        plt_items = [
            (1, len(prmtrs_with_ndr), "Prm w ND"),
            (2, len(prmtrs_wo_ndr), "Prm wo ND"),
            (4, len(bndrs_with_ndr), "Bnd w ND"),
            (5, len(bndrs_wo_ndr), "Bnd wo ND"),
            (7, len(ndrs_in_prmtrs), "ND i Prm"),
            (8, len(ndrs_out_prmtrs), "ND o Prm"),
            (10, len(ndrs_in_bndrs), "ND i Bnd"),
            (11, len(ndrs_out_bndrs), "ND o Bnd"),
        ]
        plt.bar(
            list(map(lambda x: x[0], plt_items)),
            list(map(lambda x: x[1], plt_items)),
        )
        plt.xticks(
            list(map(lambda x: x[0], plt_items)),
            list(map(lambda x: x[2], plt_items)),
        )
        plt.title(
            f"Promoters and Boundaries with and without NDR in {self._chrm.number}"
        )
        figpath = FileSave.figure_in_figdir(
            f"genes/num_prmtrs_bndrs_{bndrs}_ndr_{min_lnkr_len}_{self._chrm.number}.png"
        )

        fig, ax = plt.subplots()
        # ax.plot(x,y)
        loc = plticker.MultipleLocator(
            base=50
        )  # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)

        return figpath

    def prob_distrib_prmtr_ndrs(self) -> Path:
        lng_lnkrs = Linkers(self._chrm).ndrs()
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.with_rgn(lng_lnkrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        PlotUtil.clearfig()
        PlotUtil.show_grid()
        PlotUtil.prob_distrib(prmtrs[MEAN_C0], "Promoters")
        PlotUtil.prob_distrib(prmtrs_with_ndr[MEAN_C0], "Promoters with NDR")
        PlotUtil.prob_distrib(prmtrs_wo_ndr[MEAN_C0], "Promoters without NDR")
        plt.xlabel("Mean C0")
        plt.ylabel("Prob Distribution")
        plt.legend()
        return FileSave.figure_in_figdir("genes/prob_distrib_prmtr_ndrs.png")

    def prob_distrib_linkers_len_in_prmtrs(self) -> Path:
        PlotUtil.clearfig()
        PlotUtil.show_grid()
        linkers = Linkers(self._chrm)
        PlotUtil.prob_distrib(linkers[LEN], "linkers")
        prmtrs = Promoters(linkers.chrm)
        PlotUtil.prob_distrib(linkers.rgns_contained_in(prmtrs)[LEN], "prm linkers")
        plt.legend()
        plt.xlabel("Length")
        plt.ylabel("Prob distribution")
        plt.xlim(0, 300)
        return FileSave.figure_in_figdir(
            f"linkers/prob_distr_len_prmtrs_{self._chrm.number}.png"
        )


class MCDistribPlot:
    @classmethod
    def box_bnd_dmn_lnklen(cls):
        near = True
        pred = Prediction(35)
        mcblqs = [[], [], [], []]
        mcdl = []

        for c in YeastChrNumList:
            print(c)
            chrm = Chromosome(c, prediction=pred, spread_str=C0Spread.mcvr)
            sr = SubRegions(chrm)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            bqs = sr.bndrs.quartiles()
            if near:
                blqs = [bq.nearest_rgns(sr.lnkrs) for bq in bqs]
                dl = sr.lnkrs - blqs[0] - blqs[1] - blqs[2] - blqs[3]
            else:
                blqs = [sr.lnkrs.mid_contained_in(bq) for bq in bqs]
                dl = sr.lnkrs.mid_contained_in(sr.dmns)

            for i in range(4):
                mcblqs[i] += blqs[i][LEN].to_list()

            mcdl += dl[LEN].to_list()

        PlotUtil.font_size(20)
        bp = plt.boxplot(
            mcblqs + [mcdl],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
            boxprops={"linewidth": 3},
            medianprops={"linewidth": 3, "color": "firebrick"},
            whiskerprops={"linewidth": 3},
            capprops={"linewidth": 3},
        )
        colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:orange"]
        for box, c in zip(bp["boxes"], colors):
            box.set(
                edgecolor=c,
                fill=False,
            )

        for i, (w, cp) in enumerate(zip(bp["whiskers"], bp["caps"])):
            w.set(color=colors[int(i / 2)])
            cp.set(color=colors[int(i / 2)])

        plt.xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Domains"])
        plt.ylabel("Linker length (bp)")
        plt.tight_layout(pad=0.3, rect=(-0.06, -0.02, 1, 1))

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/lnklen_box_bnd_dmn_all{str(sr.chrm)[:-len(chrm.number)-1]}_{sr.bndrs}"
            f"{'_near' if near else ''}.svg",
            10,
            6,
        )


class DistribC0DistPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def bndrs(self, pltlim=500, dist=200):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)

        starts = np.array(range(-pltlim, pltlim, dist))
        box_starts = np.array(sr.bndrs[MIDDLE])[:, np.newaxis] + starts[np.newaxis, :]
        assert box_starts.shape == (len(sr.bndrs), math.ceil(2 * pltlim / dist))
        box_ends = box_starts + dist - 1
        pos_boxs = starts + int(dist / 2)
        c0s = []
        for s_boxs, e_boxs in zip(box_starts.T, box_ends.T):
            c0s.append(ChrmOperator(sr.chrm).c0_rgns(s_boxs, e_boxs).flatten())

        plt.boxplot(c0s, showfliers=False)
        plt.ylim(-0.5, 0.1)
        # plt.xticks(
        #     ticks=pos_boxs,
        # )
        plt.ylabel("C0 distrib")
        plt.title(
            f"C0 box distrib at distances among boundaries in chromosome {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm.id}_{str(sr.bndrs)}/"
            f"c0_box_distrib_pltlim_{pltlim}_dist_{dist}.png"
        )


class ScatterPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def scatter_c0(self) -> Path:
        PlotUtil.clearfig()
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(
            sr.non_bndry_ndrs()[LEN],
            sr.non_bndry_ndrs()[MEAN_C0],
            c="b",
            marker="s",
            label="Non-bndry NDRs",
        )
        ax1.scatter(
            sr.bndry_ndrs()[LEN],
            sr.bndry_ndrs()[MEAN_C0],
            c="r",
            marker="o",
            label="Bndry NDRs",
        )
        plt.legend(loc="upper right")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NDRS}/c0_scatter_chrm_{self._chrm.id}_ndr_{sr.min_ndr_len}"
            f"_bndrs_{sr.bndrs}.png"
        )

    def scatter_kmer(self, kmer: KMerSeq):
        PlotUtil.clearfig()
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_25)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(
            sr.non_bndry_ndrs()[LEN],
            KMer.count_w_rc(
                kmer, sr.chrm.seqf(sr.non_bndry_ndrs()[START], sr.non_bndry_ndrs()[END])
            )
            / sr.non_bndry_ndrs()[LEN],
            c="b",
            marker="s",
            label="Non-bndry NDRs",
        )
        ax1.scatter(
            sr.bndry_ndrs()[LEN],
            KMer.count_w_rc(
                kmer, sr.chrm.seqf(sr.bndry_ndrs()[START], sr.bndry_ndrs()[END])
            )
            / sr.bndry_ndrs()[LEN],
            c="r",
            marker="o",
            label="Bndry NDRs",
        )
        plt.legend(loc="upper right")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NDRS}/kmercnt_scatter_{kmer}_chrm_{self._chrm.id}_ndr_{sr.min_ndr_len}"
            f"_bndrs_{sr.bndrs}.png"
        )


class LineC0Plot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self._sr = SubRegions(self._chrm)
        self._cop = ChrmOperator(self._chrm)

    def line_c0_mean_prmtrs(self) -> Path:
        with_p = False
        with_np = False
        bnd_dmn = False
        self._sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)

        PlotUtil.clearfig()
        PlotUtil.font_size(20)
        x = np.arange(-self._sr.prmtrs._ustr_tss, self._sr.prmtrs._dstr_tss + 1)

        if with_p:
            plt.plot(
                x, self._sr.prmtrs.mean_c0(), label="Promoters", color="tab:green", lw=2
            )

        if with_np:
            nprm = Regions(self._sr.chrm, self._sr.prmtrs.complement())
            scnp = nprm.sections(self._sr.prmtrs[LEN][0])
            plt.plot(
                x,
                scnp.c0().mean(axis=0),
                label="Non-promoters",
                color="tab:orange",
                lw=2,
            )

        if bnd_dmn:
            plt.plot(
                x,
                self._sr.prmtrs_with_bndrs().mean_c0(),
                label=f"Promoters at boundaries",
                color="tab:blue",
                lw=2,
            )
            plt.plot(
                x,
                self._sr.prmtrs_wo_bndrs().mean_c0(),
                label=f"Promoters in domains",
                color="tab:red",
                lw=2,
            )

        plt.xlabel(f"Distance from TSS (bp)")
        plt.ylabel("Mean cyclizability")
        # plt.legend()
        plt.tight_layout()

        return FileSave.figure_in_figdir(
            f"{FigSubDir.PROMOTERS}/{self._chrm}_{str(self._sr.prmtrs)}/"
            f"c0_line_mean{f'_with_{self._sr.bndrs}' if bnd_dmn else ''}.png"
        )

    def line_c0_mean_lnks_nucs(self, pltlim=100) -> Path:
        nucs = False
        self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        if nucs:
            rg = self._sr.nucs
            nm = "Nucleosomes"
            d = FigSubDir.NUCLEOSOMES
        else:
            rg = self._sr.lnkrs
            nm = "Linkers"
            d = FigSubDir.LINKERS

        bl = rg.mid_contained_in(self._sr.bndrs)
        dl = rg.mid_contained_in(self._sr.dmns)
        assert len(bl) + len(dl) == len(rg)

        # c0m = self._chrm.mean_c0_around_bps(
        #     cop.in_lim(rg[MIDDLE], pltlim), pltlim, pltlim
        # )
        c0m_b = self._chrm.mean_c0_around_bps(
            self._cop.in_lim(bl[MIDDLE], pltlim), pltlim, pltlim
        )
        c0m_d = self._chrm.mean_c0_around_bps(
            self._cop.in_lim(dl[MIDDLE], pltlim), pltlim, pltlim
        )
        PlotUtil.clearfig()
        PlotUtil.font_size(20)
        x = np.arange(2 * pltlim + 1) - pltlim
        # PlotChrm(self._chrm).plot_avg()

        # plt.plot(x, c0m, label="lnk all")
        plt.plot(x, c0m_b, label=f"{nm} at boundaries", color="tab:blue", linewidth=2)
        plt.plot(x, c0m_d, label=f"{nm} in domains", color="tab:red", linewidth=2)
        plt.legend()
        PlotUtil.vertline(0, "tab:orange")
        # PlotUtil.show_grid()
        plt.xlabel(f"Distance from {str.lower(nm)} middle (bp)")
        plt.ylabel("Cyclizability (C0)")

        return FileSave.figure_in_figdir(
            f"{d}/{self._chrm}_{str(rg)}/" f"c0_line_mean_pltlim_{pltlim}.png"
        )

    def line_c0_mean_bndrs_dmns(self, pltlim=100) -> Path:
        dmns = True
        show_legend = True
        smooth = False
        self._sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
        bc0 = self._cop.c0_rgns(
            self._sr.bndrs[MIDDLE] - pltlim + 1,
            self._sr.bndrs[MIDDLE] + pltlim,
        ).mean(axis=0)

        PlotUtil.clearfig()
        PlotUtil.font_size(20)

        if dmns:
            scts = self._sr.dmns.sections(2 * pltlim)
            dc0 = self._cop.c0_rgns(scts[START], scts[END]).mean(axis=0)
        else:
            PlotChrm(self._chrm).plot_avg()

        x = np.arange(2 * pltlim) - pltlim + 1

        if smooth:
            plt.plot(x, bc0, color="tab:gray", alpha=0.5)
            plt.plot(x, gaussian_filter1d(bc0, 40), color="black")
        else:
            lw = 3
            plt.plot(x, bc0, label="Boundaries", color="dodgerblue", lw=lw)
            plt.plot(x, dc0, label="Domains", color="darkorange", lw=lw)

            if pltlim == 500:
                # PlotUtil.horizline(self._chrm.mean_c0, 'tab:green', 'avg')
                l, r = 4, 126
                rg = np.arange(l, r + 1)
                plt.plot(rg, bc0[rg + pltlim - 1], color="tab:red", lw=lw)
                # nt = np.arange(-pltlim + 1, -251)
                # plt.plot(nt, bc0[nt + pltlim - 1], color="black", lw=lw)
                # nt = np.arange(411, pltlim + 1)
                # plt.plot(nt, bc0[nt + pltlim - 1], color="black", lw=lw)
                aargs = {
                    "width": 0.0002,
                    "head_width": 0.008,
                    "head_length": 0.018,
                    "lw": lw - 1,
                    "length_includes_head": True,
                    "shape": "full",
                    "color": "red",
                    }
                plt.arrow(l, -0.154, 120, 0, **aargs)
                plt.arrow(r, -0.154, -120, 0, **aargs)
                plt.text((l + r) / 2, -0.148, "120bp", ha="center", color="red")
                

        if show_legend:
            plt.legend()

        plt.xlabel("Distance from boundary and domain section middle (bp)")
        plt.ylabel("Mean bendability")
        plt.tight_layout(pad=0.3, rect=(-0.06, -0.02, 1, 1))

        FileSave.nptxt(
            bc0,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.BOUNDARIES}/"
            f"{self._chrm}_{self._sr.bndrs}/chrm{self._chrm.id}_c0_line_mean_pltlim_{pltlim}.txt",
        )

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{self._sr.bndrs}/"
            f"c0_line_mean_pltlim_{pltlim}.svg",
            10,
            6,
        )

    def line_c0_mean_lpancs_ins(self, pltlim=100) -> Path:
        ac0 = self._cop.c0_rgns(
            self._sr.lpancrs[MIDDLE] - pltlim + 1,
            self._sr.lpancrs[MIDDLE] + pltlim,
        ).mean(axis=0)

        PlotUtil.clearfig()
        PlotUtil.font_size(20)

        scts = self._sr.lpinsds.sections(2 * pltlim)
        ic0 = self._cop.c0_rgns(scts[START], scts[END]).mean(axis=0)

        x = np.arange(2 * pltlim) - pltlim + 1
        plt.plot(x, ac0, label="Loop anchors")
        plt.plot(x, ic0, label="Loop insides")
        plt.legend()

        plt.xlabel("Distance from loop anchors and insides sections middle (bp)")
        plt.ylabel("Mean Cyclizability")
        plt.tight_layout()

        return FileSave.figure_in_figdir(
            f"{FigSubDir.LOOP_ANCHORS}/{self._chrm}_{self._sr.lpancrs}/"
            f"c0_line_mean_pltlim_{pltlim}.png"
        )

    def line_c0_bndrs_q(self, pltlim=100):
        self._sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
        bqs = self._sr.bndrs.quartiles()
        bqc0 = [
            self._cop.c0_rgns(
                bq[MIDDLE] - pltlim + 1,
                bq[MIDDLE] + pltlim,
            ).mean(axis=0)
            for bq in bqs
        ]
        x = np.arange(2 * pltlim) - pltlim + 1
        for y, l in zip(bqc0, ["Q1", "Q2", "Q3", "Q4"]):
            plt.plot(x, y, label=l)

        plt.legend()
        plt.xlabel("Distance from boundary and domain sections middle (bp)")
        plt.ylabel("Mean Cyclizability")
        plt.tight_layout()

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{self._sr.bndrs}/"
            f"c0_line_mean_q_pltlim_{pltlim}.png"
        )

    def line_c0_bndrs_indiv_toppings(self) -> None:
        self._sr.bsel = BndSel(BoundariesType.FANCN, BndFParm.SHR_50)
        for bndrs, pstr in zip(
            [self._sr.bnds_in_prms(), self._sr.bnds_in_nprms()], ["prmtr", "nonprmtr"]
        ):
            for bndry in bndrs:
                self._line_c0_bndry_indiv_toppings(bndry, str(bndrs), pstr)

    def _line_c0_bndry_indiv_toppings(
        self, bndry: pd.Series, bstr: str, pstr: str
    ) -> Path:
        self.line_c0_toppings(
            getattr(bndry, START) - 100, getattr(bndry, END) + 100, save=False
        )
        plt.title(
            f"C0 around {pstr} boundary at {getattr(bndry, MIDDLE)} bp of chrm {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{bstr}/"
            f"bndry_{pstr}_{getattr(bndry, START)}_{getattr(bndry, END)}_score_"
            f"{round(getattr(bndry, SCORE), 2)}.png"
        )

    def line_c0_prmtrs_indiv_toppings(self) -> None:
        prmtrs = Promoters(self._chrm)
        for prmtr in prmtrs:
            self._line_c0_prmtr_indiv_toppings(prmtr)

    def _line_c0_prmtr_indiv_toppings(self, prmtr: pd.Series) -> Path:
        self.line_c0_toppings(getattr(prmtr, START), getattr(prmtr, END), save=False)
        fr = "frw" if getattr(prmtr, STRAND) == 1 else "rvs"
        plt.title(f"C0 in {fr} promoter {getattr(prmtr, START)}-{getattr(prmtr, END)}")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.PROMOTERS}/{self._chrm.id}/"
            f"{fr}_{getattr(prmtr, START)}_{getattr(prmtr, END)}.png"
        )

    def line_c0_lpancs_indiv_toppings(self) -> None:
        for lpanc in self._sr.lpancrs:
            self._line_c0_lpancs_indiv_toppings(lpanc, str(self._sr.lpancrs))

    def _line_c0_lpancs_indiv_toppings(self, lpanc: pd.Series, lpstr: str) -> Path:
        self.line_c0_toppings(getattr(lpanc, START), getattr(lpanc, END), save=False)
        return FileSave.figure_in_figdir(
            f"{FigSubDir.LOOP_ANCHORS}/{self._chrm}_{lpstr}/"
            f"{getattr(lpanc, START)}_{getattr(lpanc, END)}.png"
        )

    def line_c0_toppings(
        self, start: PosOneIdx, end: PosOneIdx, show: bool = False, save: bool = True
    ) -> Path:
        def _within_bool(pos: pd.Series) -> pd.Series:
            return (start <= pos) & (pos <= end)

        def _within(pos: pd.Series) -> pd.Series:
            return pos.loc[(start <= pos) & (pos <= end)]

        def _end_within(rgns: Regions) -> Regions:
            return rgns[(_within_bool(rgns[START])) | (_within_bool(rgns[END]))]

        def _clip(starts: pd.Series, ends: pd.Series) -> tuple[pd.Series, pd.Series]:
            scp = starts.copy()
            ecp = ends.copy()
            scp.loc[scp < start] = start
            ecp.loc[ecp > end] = end
            return scp, ecp

        def _vertline(pos: pd.Series, color: str) -> None:
            for p in _within(pos):
                PlotUtil.vertline(p, color=color)

        gnd = self._chrm.mean_c0

        def _nuc_ellipse(dyads: pd.Series, clr: str) -> None:
            for d in dyads:
                ellipse = Ellipse(
                    xy=(d, gnd),
                    width=146,
                    height=0.2,
                    edgecolor=clr,
                    fc="None",
                    lw=1,
                )
                plt.gca().add_patch(ellipse)

        def _bndrs(mids: pd.Series, scr: pd.Series, clr: str) -> None:
            wb = 50
            hb = 0.1
            for m, s in zip(mids, scr):
                points = [
                    [m - wb * s, gnd + hb * s],
                    [m, gnd],
                    [m + wb * s, gnd + hb * s],
                ]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=2)
                plt.gca().add_patch(line)

        def _lng_linkrs(lnks: Linkers, clr: str) -> None:
            sc, ec = _clip(lnks[START], lnks[END])
            for s, e in zip(sc, ec):
                rectangle = plt.Rectangle(
                    (s, gnd - 0.05), e - s, 0.1, fc=clr, alpha=0.5
                )
                plt.gca().add_patch(rectangle)

        def _tss(tss: pd.Series, frw: bool, clr: str) -> None:
            diru = 1 if frw else -1
            for t in tss:
                points = [[t, gnd], [t, gnd + 0.15], [t + 50 * diru, gnd + 0.15]]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=3)
                plt.gca().add_patch(line)

        def _text() -> None:
            for x, y in self._text_pos_calc(start, end, 0.1):
                plt.text(x, gnd + y, self._chrm.seq[x - 1 : x + 3], fontsize="xx-small")

        PlotUtil.clearfig()
        PlotUtil.show_grid(which="both")
        pltchrm = PlotChrm(self._chrm)
        pltchrm.line_c0(start, end)
        dyads = self._sr.nucs[MIDDLE]

        colors = ["tab:orange", "tab:brown", "tab:purple", "tab:green"]
        labels = ["nuc", "bndrs", "tss", "lng lnk"]
        _nuc_ellipse(_within(dyads), colors[0])
        _bndrs(_within(self._sr.bndrs[MIDDLE]), self._sr.bndrs[SCORE], colors[1])
        _tss(_within(self._sr.genes.frwrd_genes()[START]), True, colors[2])
        _tss(_within(self._sr.genes.rvrs_genes()[END]), False, colors[2])
        _lng_linkrs(_end_within(self._sr.ndrs), colors[3])
        _text()

        PlotUtil.legend_custom(colors, labels)

        if show:
            plt.show()

        if save:
            return FileSave.figure_in_figdir(
                f"{FigSubDir.CROSSREGIONS}/line_c0_toppings.png"
            )

    def _text_pos_calc(
        self, start: PosOneIdx, end: PosOneIdx, amp: float
    ) -> Iterator[tuple[PosOneIdx, float]]:
        return zip(
            range(start, end, 4),
            [amp, amp / 2, -amp / 2, -amp] * math.ceil((end - start) / 4 / 4),
        )


class PaperLineC0Plot:
    @classmethod
    def bnd(cls):
        chrm = Chromosome("VL", spread_str=C0Spread.mcvr)
        cop = ChrmOperator(chrm)
        sr = SubRegions(chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        # 341 - 481 linker, 401 - 600 boundary
        # s, e =  303341, 303600
        # ls, le = 303341, 303481
        # bs, be = 303401, 303600
        # lg = -0.4
        # bg = -0.5
        s, e = 59201, 59476
        ls, le = 59339, 59476
        bs, be = 59201, 59400
        lg = -0.2
        bg = -0.3
        c0 = cop.c0(s, e)
        x = np.arange(s, e + 1)
        PlotUtil.font_size(20)
        plt.plot(x, c0)
        plt.xlabel("Position in bp")
        plt.ylabel("Cyclizability")

        line = plt.Polygon(
            [[ls, lg], [le, lg]], closed=None, fill=None, edgecolor="tab:blue", lw=3
        )
        plt.gca().add_patch(line)
        plt.text(
            (ls + le) / 2,
            lg,
            "linker",
            color="tab:blue",
            ha="center",
            va="top",
        )

        line = plt.Polygon(
            [[bs, bg], [be, bg]], closed=None, fill=None, edgecolor="tab:red", lw=3
        )
        plt.gca().add_patch(line)
        plt.text(
            (bs + be) / 2,
            bg,
            "boundary",
            color="tab:red",
            ha="center",
            va="top",
        )

        FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{chrm}_{sr.bndrs}/bndrs_{s}_{e}.png"
        )


class MCLineC0Plot:
    pred = Prediction(35)

    @classmethod
    def line_c0_mean_prmtrs(cls):
        with_p = False
        with_np = False
        bnd_dmn = True
        pc0 = []
        npc0 = []
        pbc0 = []
        pdc0 = []
        for c in YeastChrNumList:
            print(c)
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            if with_p:
                pc0.append(
                    np.full((len(sr.prmtrs), sr.prmtrs[LEN][0]), sr.prmtrs.mean_c0())
                )
            if with_np:
                nprm = Regions(sr.chrm, sr.prmtrs.complement())
                scnp = nprm.sections(sr.prmtrs[LEN][0])
                npc0.append(np.full((len(scnp), scnp[LEN][0]), scnp.c0().mean(axis=0)))

            if bnd_dmn:
                b, d = sr.prmtrs_with_bndrs(), sr.prmtrs_wo_bndrs()
                pbc0.append(np.full((len(b), b[LEN][0]), b.mean_c0()))
                pdc0.append(np.full((len(d), d[LEN][0]), d.mean_c0()))

        x = np.arange(-sr.prmtrs._ustr_tss, sr.prmtrs._dstr_tss + 1)
        PlotUtil.font_size(20)

        lw = 3
        if with_p:
            pmc0 = np.vstack(pc0).mean(axis=0)
            plt.plot(x, pmc0, color="tab:blue", label="Promoters", lw=lw)

        if with_np:
            npmc0 = np.vstack(npc0).mean(axis=0)
            plt.plot(x, npmc0, color="tab:orange", label="Non-promoters", lw=lw)

        if bnd_dmn:
            mc0b = np.vstack(pbc0).mean(axis=0)
            mc0d = np.vstack(pdc0).mean(axis=0)
            plt.plot(x, mc0b, color="dodgerblue", label="Promoters at boundaries", lw=lw)
            plt.plot(x, mc0d, color="darkorange", label="Promoters in domains", lw=lw)

        # PlotUtil.vertline(0, "tab:orange", linewidth=2)
        plt.legend()
        plt.xlabel(f"Distance from TSS (bp)")
        plt.ylabel("Mean bendability")
        plt.tight_layout(pad=0.3, rect=(-0.06, -0.02, 1, 1))

        return FileSave.figure_in_figdir(
            f"{FigSubDir.PROMOTERS}/all{str(sr.chrm)[:-len(sr.chrm.number)-1]}_{sr.prmtrs}/"
            f"c0_line_mean{f'_with_{sr.bndrs}' if bnd_dmn else '' }.svg",
            10,
            6,
        )

    @classmethod
    def line_c0_mean_lnks_nucs(cls, pltlim=100):
        nucs = True
        mcb_c0 = []
        mcd_c0 = []
        for c in YeastChrNumList:
            print(c)
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
            cop = ChrmOperator(chrm)

            b, d = (
                (sr.nucs_in_bnds(), sr.nucs_in_dmns())
                if nucs
                else (sr.lnks_in_bnds(), sr.lnks_in_dmns())
            )

            mcb_c0.append(
                cop.c0_rgns(
                    cop.in_lim(b[MIDDLE], pltlim) - pltlim,
                    cop.in_lim(b[MIDDLE], pltlim) + pltlim,
                )
            )
            mcd_c0.append(
                cop.c0_rgns(
                    cop.in_lim(d[MIDDLE], pltlim) - pltlim,
                    cop.in_lim(d[MIDDLE], pltlim) + pltlim,
                )
            )

        mc0b = np.vstack(mcb_c0).mean(axis=0)
        mc0d = np.vstack(mcd_c0).mean(axis=0)

        x = np.arange(2 * pltlim + 1) - pltlim
        PlotUtil.font_size(20)
        labels = (
            ["Nucleosomes at boundaries", "Nucleosomes in domains"]
            if nucs
            else ["Linkers at boundaries", "Linkers in domains"]
        )
        plt.plot(x, mc0b, color="tab:blue", label=labels[0], linewidth=2)
        plt.plot(x, mc0d, color="tab:red", label=labels[1], linewidth=2)
        PlotUtil.vertline(0, "tab:orange", linewidth=2)
        plt.legend()
        plt.xlabel(f"Distance from {'nucleosome' if nucs else 'linker'} middle (bp)")
        plt.ylabel("Cyclizability (C0)")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NUCLEOSOMES if nucs else FigSubDir.LINKERS}/c0_line_mean_all{str(sr.chrm)[:-4]}_{sr.bndrs}_pltlim_{pltlim}.png"
        )

    @classmethod
    def _sr(cls, c: str):
        chrm = Chromosome(c, prediction=cls.pred, spread_str=C0Spread.mcvr)
        return SubRegions(chrm), chrm

    @classmethod
    def line_c0_mean_bndrs(cls, pltlim=100):
        bc0 = []
        dc0 = []
        l = []
        mc0 = []
        for c in YeastChrNumList:
            print(c)
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            bc0.append(
                ChrmOperator(chrm).c0_rgns(
                    (sr.bndrs[MIDDLE] - pltlim + 1).tolist(),
                    (sr.bndrs[MIDDLE] + pltlim).tolist(),
                )
            )
            dc0.append(sr.dmns.sections(2 * pltlim).c0())
            l.append(chrm.total_bp)
            mc0.append(chrm.mean_c0)

        mc0b = np.vstack(bc0).mean(axis=0)
        mc0d = np.vstack(dc0).mean(axis=0)
        mc0 = np.dot(l, mc0) / sum(l)

        FileSave.nptxt(
            mc0b,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.BOUNDARIES}/"
            f"c0_line_mean_all{str(sr.chrm)[:-len(chrm.number)-1]}_{sr.bndrs}_pltlim_{pltlim}.txt",
        )

        PlotUtil.font_size(20)
        lw = 3
        x = np.arange(2 * pltlim) - pltlim + 1
        plt.plot(x, mc0b, label="Boundaries", color="dodgerblue", lw=lw)
        plt.plot(x, mc0d, label="Domains", color="darkorange", lw=lw)
        if pltlim == 500:
            # PlotUtil.horizline(mc0, 'darkgreen', 'avg')
            rg = np.arange(-41, 80)
            plt.plot(rg, mc0b[rg + pltlim - 1], color="red", lw=lw)
            aargs = {
                "width": 0.0002,
                "head_width": 0.004,
                "head_length": 0.01,
                "lw": lw - 1,
                "length_includes_head": True,
                "shape": "full",
                "color": "red",
            }
            plt.arrow(-41, -0.198, 120, 0, **aargs)
            plt.arrow(79, -0.198, -120, 0, **aargs)
            plt.text(20, -0.196, "120bp", ha="center", color="red")
            # nt = np.arange(-pltlim + 1, -326)
            # plt.plot(nt, mc0b[nt + pltlim - 1], color="black", lw=lw)
            # nt = np.arange(370, pltlim + 1)
            # plt.plot(nt, mc0b[nt + pltlim - 1], color="black", lw=lw)

        plt.legend()
        plt.xlabel("Distance from boundary and domain section middle (bp)")
        plt.ylabel("Mean bendability")
        plt.tight_layout(pad=0.3, rect=(-0.06, -0.02, 1, 1))

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/c0_line_mean_all{str(sr.chrm)[:-len(sr.chrm.number)-1]}_{sr.bndrs}_pltlim_{pltlim}.svg",
            10,
            6,
        )

    @classmethod
    def line_c0_bndrs_q(cls, pltlim=100):
        mcbqc0 = [[], [], [], []]
        for c in YeastChrNumList:
            print(c)
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            bqs = sr.bndrs.quartiles()
            bqc0 = [
                ChrmOperator(chrm).c0_rgns(
                    bq[MIDDLE] - pltlim + 1,
                    bq[MIDDLE] + pltlim,
                )
                for bq in bqs
            ]
            for i in range(4):
                mcbqc0[i].append(bqc0[i])

        x = np.arange(2 * pltlim) - pltlim + 1
        for i in range(4):
            mcbqc0[i] = np.vstack(mcbqc0[i]).mean(axis=0)

        PlotUtil.font_size(20)
        for y, l, c in zip(
            mcbqc0,
            ["Q1 (strong)", "Q2", "Q3", "Q4 (weak)"],
            ["tab:green", "tab:purple", "tab:brown", "tab:pink"],
        ):
            plt.plot(x, y, label=l, lw=3, color=c)

        plt.legend()
        plt.xlabel("Distance from boundary middle (bp)")
        plt.ylabel("Mean bendability")
        plt.tight_layout(pad=0.3, rect=(-0.05, -0.02, 1, 1))

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/c0_line_mean_q_all{str(sr.chrm)[:-len(sr.chrm.number)-1]}_{sr.bndrs}_pltlim_{pltlim}.svg",
            10,
            6,
        )


class MCMotifsM35:
    @classmethod
    def enr_line(cls):
        plt_some = True
        plt_indiv = False
        plt_all = False

        r = "b"
        if r == "b":
            d = GDataSubDir.BOUNDARIES
            yl = (0, 1.0)
        elif r == "n":
            d = GDataSubDir.NUCLEOSOMES
            yl = (0, 15)

        pred = Prediction(35)
        nums = YeastChrNumList
        enrr = [[] for _ in range(len(nums))]
        lb = []
        for i, c in enumerate(nums):
            print(c)
            chrm = Chromosome(c, pred, C0Spread.mcvr)
            mt = MotifsM35(c)
            sr = SubRegions(chrm)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_WD_100)
            if r == "b":
                rgns = sr.bndrs
            elif r == "n":
                rgns = sr.nucs

            for m in range(N_MOTIFS):
                enrr[i].append(mt.enr_rgns(m, rgns[START], rgns[END]).mean(axis=0))

            lb.append(len(rgns))

        enrrm = (np.array(lb)[:, np.newaxis, np.newaxis] * np.array(enrr)).mean(axis=0)
        assert enrrm.shape == (N_MOTIFS, rgns[END][0] - rgns[START][0] + 1)

        x = np.arange(enrrm.shape[1]) - enrrm.shape[1] // 2

        if plt_some and r == "b":
            h = [50, 54, 254, 55, 85]
            l = [111, 131, 29, 15, 77]

            def _p(k):
                PlotUtil.font_size(16)
                fig, axes = plt.subplots(
                    5, 1, sharex="all", sharey="all", constrained_layout=True
                )
                plt.ylim(0, 0.70)
                for i, m in enumerate(k):
                    ax = axes[i]
                    ax.plot(x, enrrm[m])

                fig.supxlabel("Position (bp)")
                fig.supylabel("Matching score")

                FileSave.figure_in_figdir(
                    f"{d}/all{str(chrm)[:-len(chrm.number)-1]}_{rgns}/motif_m35_v4/line_enr_some_{k[0]}.svg",
                    3,
                    8,
                )

            _p(h)
            _p(l)

        if plt_indiv:
            PlotUtil.font_size(20)
            for m in range(N_MOTIFS):
                PlotUtil.clearfig()
                plt.ylim(*yl)
                plt.plot(x, enrrm[m], color="k")
                plt.xlabel("Position (bp)")
                plt.ylabel("Matching score")
                FileSave.figure_in_figdir(
                    f"{d}/all{str(chrm)[:-len(chrm.number)-1]}_{rgns}/motif_m35_v4/line_enr_{m}.png",
                    8,
                    8,
                )

        if plt_all:
            PlotUtil.font_size(12)
            PlotUtil.clearfig()
            fig, axes = plt.subplots(16, 16, sharex="all", sharey="all")
            for m in range(N_MOTIFS):
                ax = axes[m // 16, m % 16]
                ax.plot(x, enrrm[m])
                ax.annotate(str(m), xy=(0.75, 0.85), xycoords="axes fraction")

            return FileSave.figure_in_figdir(
                f"{d}/all{str(chrm)[:-len(chrm.number)-1]}_{rgns}/motif_m35_v4/line_enr_all.png",
                24,
                24,
            )

    @classmethod
    def enr_compare(cls):
        enra = np.array([], dtype=np.float16).reshape(N_MOTIFS, 0)
        enrb = np.array([], dtype=np.float16).reshape(N_MOTIFS, 0)
        pred = Prediction(35)
        for c in YeastChrNumList:
            print(c)
            chrm = Chromosome(c, pred, C0Spread.mcvr)
            sr = SubRegions(chrm)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            mt = MotifsM35(c)
            b = mt._score[:, sr.bndrs.cover_mask]
            d = mt._score[:, sr.dmns.cover_mask]
            enra = np.hstack((enra, b.astype(np.float16)))
            enrb = np.hstack((enrb, d.astype(np.float16)))

        FileSave.npy(
            enra,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.BOUNDARIES}/"
            f"all{str(chrm)[:-len(chrm.number)-1]}_{sr.bndrs}/motif_m35_v{mt._V}/match_score_float16.npy",
        )
        FileSave.npy(
            enrb,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.DOMAINS}/"
            f"all{str(chrm)[:-len(chrm.number)-1]}_{sr.bndrs}/motif_m35_v{mt._V}/match_score_float16.npy",
        )
        z = [(i,) + ztest(enra[i], enrb[i]) for i in range(N_MOTIFS)]
        df = pd.DataFrame(z, columns=[MOTIF_NO, ZTEST_VAL, P_VAL])

        dn = f"{GDataSubDir.BOUNDARIES}/all{str(chrm)[:-len(chrm.number)-1]}_{sr.bndrs}/motif_m35_v{mt._V}"
        fn = f"enr_comp_{sr.dmns}"
        FileSave.tsv_gdatadir(df, f"{dn}/{fn}.tsv", precision=-1)
        FileSave.tsv_gdatadir(
            df.sort_values(ZTEST_VAL), f"{dn}/sorted_{fn}.tsv", precision=-1
        )

    @classmethod
    def combine_z(cls):
        pred = Prediction(35)
        lb = []
        z = np.zeros((N_MOTIFS,))
        p = np.zeros((N_MOTIFS,))
        for c in YeastChrNumList:
            print(c)
            chrm = Chromosome(c, pred, C0Spread.mcvr)
            sr = SubRegions(chrm)
            sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
            rega, regb, d = sr.bndrs, sr.dmns, GDataSubDir.BOUNDARIES
            sl = math.sqrt(len(rega))
            df = pd.read_csv(
                f"{PathObtain.gen_data_dir()}/{d}/{chrm}_{rega}/motif_m35_v4/"
                f"enr_comp_{regb}.tsv",
                sep="\t",
            )
            z += df[ZTEST_VAL] * sl
            p += df[P_VAL] * sl
            lb.append(sl)

        z /= sum(lb)
        p /= sum(lb)
        df = pd.DataFrame({MOTIF_NO: range(N_MOTIFS), ZTEST_VAL: z, P_VAL: p})
        dn = f"{d}/all{str(chrm)[:-len(chrm.number)-1]}_{rega}"
        FileSave.tsv_gdatadir(df, f"{dn}/enr_comp_comb_{regb}.tsv", precision=-1)
        FileSave.tsv_gdatadir(
            df.sort_values(ZTEST_VAL, ignore_index=True),
            f"{dn}/sorted_enr_comp_comb_{regb}.tsv",
            precision=-1,
        )


class SegmentLineC0Plot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def sl_lnkrs(self):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        lnks = [sr.lnkrs.len_in(mn=st, mx=st + 20 - 1) for st in (11, 31, 51, 71)]
        lnksb = [lnk.mid_contained_in(sr.bndrs) for lnk in lnks]
        lnksd = [lnk - lnkb for lnk, lnkb in zip(lnks, lnksb)]
        lnksbc0 = [
            np.array([resize(c0_arr, (101,)) for c0_arr in lnkb.c0()]).mean(axis=0)
            for lnkb in lnksb
        ]
        lnksdc0 = [
            np.array([resize(c0_arr, (101,)) for c0_arr in lnkd.c0()]).mean(axis=0)
            for lnkd in lnksd
        ]
        fig, ax = plt.subplots(4, 2)
        for i in range(4):
            ax[i][0].plot(lnksbc0[i])
            ax[i][1].plot(lnksdc0[i])

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/linkers/lnkrs_c0_segm_bndrs_dmns_{self._chrm}.png"
        )


RIGID_PAIRS = [
    ("GC", "TT"),
    ("AA", "GC"),
    ("GC", "TA"),
    ("AT", "GC"),
    ("AA", "CG"),
    ("AA", "CC"),
    ("GG", "TT"),
    ("CG", "TT"),
    ("TT", "CC"),
    ("TA", "CC"),
]

FLEXIBLE_PAIRS = [
    ("CC", "CC"),
    ("GC", "CG"),
    ("GC", "CC"),
    ("TA", "TT"),
    ("AT", "TT"),
    ("AA", "TA"),
    ("AA", "AA"),
    ("AA", "TT"),
    ("TT", "TT"),
    ("GC", "GC"),
]


class LinePlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def helsep_mean_bndrs(self, pltlim=100):
        rigid = False

        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        hs = HelSep()
        harr = np.empty((len(sr.bndrs), 2 * pltlim + 1))
        for i, b in enumerate(sr.bndrs):
            dfh = hs.helical_sep_of(
                list(
                    map(
                        lambda p: sr.chrm.seqf(
                            p - SEQ_LEN / 2,
                            p + SEQ_LEN / 2 - 1,
                        ),
                        range(
                            getattr(b, MIDDLE) - pltlim, getattr(b, MIDDLE) + pltlim + 1
                        ),
                    )
                )
            )
            pairs = RIGID_PAIRS if rigid else FLEXIBLE_PAIRS
            cols = [DincUtil.pair_str(*p) for p in pairs]
            harr[i] = np.sum(dfh[cols].to_numpy(), axis=1)

        harr = harr.mean(axis=0)
        x = np.arange(2 * pltlim + 1) - pltlim
        pt = "rigid" if rigid else "flexible"
        PlotUtil.show_grid()
        plt.plot(x, harr)
        plt.xlabel("Position from boundary mid")
        plt.ylabel(f"Helsep {pt} content")
        plt.title(f"Helsep {pt} content in bndrs")
        return FileSave.figure_in_figdir(
            f"{sr.bndrs.fig_subdir()}/helsep_{pt}_content_pltlim_{pltlim}.png"
        )

    def kmer_mean_tss(self, kmer: str):
        al = True
        rc = True
        smt = 10
        ma = 20
        u, d = 599, 400
        pred = Prediction(35)
        if al:
            cl = [Chromosome(c, pred, C0Spread.mcvr) for c in YeastChrNumList]
        else:
            cl = [self._chrm]

        arrs = []

        for ch in cl:
            print(ch.id)
            prmtrs = Promoters(ch, ustr_tss=u, dstr_tss=d)
            rgns = prmtrs
            seqs = ch.seqf(rgns[START], rgns[END])
            arr = np.zeros((len(seqs), len(seqs[0])))
            pos_func = KMer.find_pos_w_rc if rc else KMer.find_pos
            for i, seq in enumerate(seqs):
                arr[i, pos_func(kmer, seq)] = 1

            arr = Promoters.val_tss_align(arr, prmtrs[STRAND])
            arrs.append(arr)

        marr = np.vstack(arrs).mean(axis=0)
        marr = ChrmCalc.moving_avg(marr, ma)
        PlotUtil.clearfig()
        PlotUtil.font_size(20)
        x = np.arange(u + d + 1) - u
        x = x[ma // 2 : len(x) - ((ma - 1) // 2)]
        if smt > 0:
            plt.plot(x, marr, color="lightskyblue", alpha=0.5, lw=2, ls="-")
            plt.plot(x, gaussian_filter1d(marr, smt), color="tab:blue", lw=3)
        else:
            plt.plot(x, marr, color="tab:blue", lw=3)
        plt.xlabel("Distance from TSS (bp)")
        plt.ylabel(f"{kmer}{f'/{rev_comp(kmer)}' if rc else ''} content")
        plt.tight_layout()
        dr = (
            f"{GDataSubDir.PROMOTERS}/all{str(cl[-1])[:-len(cl[-1].number)-1]}_{prmtrs}"
            if al
            else rgns.fig_subdir()
        )
        return FileSave.figure_in_figdir(
            f"{dr}/kmer/{kmer}_content{'_rc' if rc else ''}"
            f"_ma_{ma}{f'_smt_{smt}' if smt > 0 else ''}.png"
        )

    def kmer_mean_rgns(self, kmer: str, plim=100):
        rc = True
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_100)
        rgns = sr.nucs
        mids = ChrmOperator(self._chrm).in_lim(rgns[MIDDLE], plim)
        seqs = sr.chrm.seqf(mids - plim, mids + plim)
        arr = np.zeros((len(seqs), len(seqs[0])))
        pos_func = KMer.find_pos_w_rc if rc else KMer.find_pos
        for i, seq in enumerate(seqs):
            arr[i, pos_func(kmer, seq)] = 1

        arr = arr.mean(axis=0)
        x = np.arange(2 * plim + 1) - plim
        PlotUtil.clearfig()
        plt.plot(x, arr)
        plt.xlabel("Position from mid")
        plt.ylabel(f"{kmer} content")

        return FileSave.figure_in_figdir(
            f"{rgns.fig_subdir()}/kmer/{kmer}_content_plim_{plim}{'_rc' if rc else ''}.png"
        )


class MCLinePlot:
    @classmethod
    def kmer_c0_tss(self):
        kmer = "GAAGAGC"

        rc = True
        smt = 10
        ma = 20
        u, d = 599, 400
        pred = Prediction(35)

        cl = [Chromosome(c, pred, C0Spread.mcvr) for c in YeastChrNumList]
        prmtrs = None

        arrs = []
        pc0 = []

        for ch in cl:
            print(ch.id)
            prmtrs = Promoters(ch, ustr_tss=u, dstr_tss=d)
            rgns = prmtrs
            seqs = ch.seqf(rgns[START], rgns[END])
            arr = np.zeros((len(seqs), len(seqs[0])))
            pos_func = KMer.find_pos_w_rc if rc else KMer.find_pos
            for i, seq in enumerate(seqs):
                arr[i, pos_func(kmer, seq)] = 1

            arr = Promoters.val_tss_align(arr, prmtrs[STRAND])
            arrs.append(arr)
            pc0.append(np.full((len(prmtrs), prmtrs[LEN][0]), prmtrs.mean_c0()))

        marr = np.vstack(arrs).mean(axis=0)
        marr = ChrmCalc.moving_avg(marr, ma)
        PlotUtil.clearfig()
        PlotUtil.font_size(24)
        fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
        x = np.arange(u + d + 1) - u
        x = x[ma // 2 : -((ma - 1) // 2)]

        axs[0].plot(x, marr, color="lightskyblue", alpha=0.5, lw=2, ls="-")
        axs[0].plot(x, gaussian_filter1d(marr, smt), color="tab:blue", lw=3)
        axs[0].set_ylabel(f"{kmer}{f'/{rev_comp(kmer)}' if rc else ''} content")

        x = np.arange(-u, d + 1)
        x = x[ma // 2 : -((ma - 1) // 2)]

        pmc0 = np.vstack(pc0).mean(axis=0)
        pmc0 = pmc0[ma // 2 : -((ma - 1) // 2)]

        axs[1].plot(x, pmc0, color="tab:blue", label="Promoters", lw=3)

        axs[1].set_ylabel("Mean cyclizability")

        def _con(x):
            con = ConnectionPatch(
                xyA=(x, -0.24),
                xyB=(x, 0.00028),
                coordsA="data",
                coordsB="data",
                axesA=axs[1],
                axesB=axs[0],
                color="tomato",
            )
            con.set(lw=3, ls="--")
            return con

        axs[1].add_artist(_con(20))
        axs[1].add_artist(_con(-200))

        axs[1].set_xlabel("Distance from TSS (bp)")

        dr = (
            f"{GDataSubDir.PROMOTERS}/all{str(cl[-1])[:-len(cl[-1].number)-1]}_{prmtrs}"
        )

        return FileSave.figure_in_figdir(
            f"{dr}/kmer/{kmer}_content{'_rc' if rc else ''}_c0"
            f"_ma_{ma}{f'_smt_{smt}' if smt > 0 else ''}.png",
            12,
            12,
        )


class PlotPrmtrsBndrs:
    WB_DIR = "with_boundaries"
    WOB_DIR = "without_boundaries"
    BOTH_DIR = "both"

    def __init__(self):
        pass

    def helsep_box(self) -> Path:
        pass

    def dinc_explain_scatter(self) -> Path:
        sr = SubRegions(Chromosome("VL"))
        mc0 = sr.prmtrs[MEAN_C0]

        ta = sum(KMer.count("TA", sr.chrm.seqf(sr.prmtrs[START], sr.prmtrs[END])))
        cg = sum(KMer.count("CG", sr.chrm.seqf(sr.prmtrs[START], sr.prmtrs[END])))
        fig, axs = plt.subplots(2)
        fig.suptitle("Scatter plot of mean C0 vs TpA and CpG content in promoters")
        axs[0].scatter(mc0, ta)
        axs[1].scatter(mc0, cg)
        axs[0].set_ylabel("TpA content")
        axs[0].set_xlabel("Mean C0")
        axs[1].set_ylabel("CpG content")
        axs[1].set_xlabel("Mean C0")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/prmtrs_ta_cg_scatter.png"
        )

    def dinc_explain_box(self) -> Path:
        subr = SubRegions(Chromosome("VL"))
        subr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR_50)
        pmwb = subr.prmtrs_with_bndrs()
        pmob = subr.prmtrs_wo_bndrs()
        labels = ["Prmtrs w b", "Prmtrs wo b"]
        fig, axs = plt.subplots(3)
        fig.suptitle("TpA and CpG content in promoters")
        for axes in axs:
            axes.grid(which="both")
        PlotUtil.box_many(
            [pmwb[MEAN_C0], pmob[MEAN_C0]],
            labels=labels,
            ylabel="Mean C0",
            pltobj=axs[0],
        )
        for dinc, axes in zip(
            ["TA", "CG"],
            axs[1:],
        ):
            PlotUtil.box_many(
                [
                    sum(KMer.count(dinc, subr.chrm.seqf(pmwb[START], pmwb[END]))),
                    sum(KMer.count(dinc, subr.chrm.seqf(pmob[START], pmob[END]))),
                ],
                labels=labels,
                ylabel=f"{dinc} content",
                pltobj=axes,
            )

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/dinc_explain_VL.png"
        )

    def _mean_dinc(self, rgns: Regions, cnt_fnc: Callable) -> NDArray[(Any,), float]:
        return self._total_dinc(rgns, cnt_fnc) / rgns[LEN].to_numpy()

    def both_sorted_motif_contrib(self):
        for i, num in enumerate(MotifsM30().sorted_contrib()):
            self._both_motif_contrib_single("both_sorted_motif", num, i)

    def both_motif_contrib(self):
        for num in range(256):
            self._both_motif_contrib_single(self.BOTH_DIR, num)

    def _both_motif_contrib_single(
        self, bthdir: str, num: int, srtidx: int = None
    ) -> Path:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(plt.imread(self._contrib_file(self.WB_DIR, num, "png")))
        axs[0].set(title="with boundaries")
        axs[1].imshow(plt.imread(self._contrib_file(self.WOB_DIR, num, "png")))
        axs[1].set(title="without boundaries")
        fig.suptitle(f"Contrib of motif {num} in promoters")
        return FileSave.figure(
            self._contrib_file(
                bthdir, f"{srtidx}_{num}" if srtidx is not None else num, "png"
            )
        )

    def _contrib_file(
        self, dir: str, num: int | str, frmt: Literal["svg"] | Literal["png"]
    ) -> str:
        return (
            f"{PathObtain.figure_dir()}/{FigSubDir.PROMOTERS}/"
            f"distribution_around_promoters/{dir}/motif_{num}.{frmt}"
        )

    def svg2png_contrib(self):
        for i in range(256):
            svg2png(
                url=self._contrib_file(self.WOB_DIR, i, "svg"),
                write_to=self._contrib_file(self.WOB_DIR, i, "png"),
            )
