from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from re import S
from typing import Iterable, Literal, NamedTuple, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptyping import NDArray

from chromosome.chromosome import Chromosome, MultiChrm, PlotChrm
from chromosome.genes import STRAND, Promoters
from chromosome.nucleosomes import Linkers
from chromosome.regions import (
    Regions,
    RegionsInternal,
    START,
    END,
    MEAN_C0,
    MIDDLE,
    LEN,
)
from models.prediction import Prediction
from util.util import FileSave, PlotUtil, PathObtain, Attr
from util.custom_types import ChrId, PositiveInt, PosOneIdx, ZeroToOne
from util.constants import ChrIdList, FigSubDir, GDataSubDir


SCORE = "score"
SCORE_MEDIAN_V_RES_200 = -0.22432


class BoundaryNT(NamedTuple):
    left: PosOneIdx
    right: PosOneIdx
    middle: PosOneIdx
    score: float


class BndParmT(TypedDict, total=False):
    res: int
    lim: int
    score_perc: float


class BndParm:
    HIRS_WD_50 = BndParmT(res=200, lim=250, score_perc=0.5)
    HIRS_WD_100 = BndParmT(res=200, lim=250, score_perc=1.0)
    HIRS_WD_25 = BndParmT(res=200, lim=250, score_perc=0.25)
    HIRS_SHR_50 = BndParmT(res=200, lim=100, score_perc=0.5)
    HIRS_SHR_100 = BndParmT(res=200, lim=100, score_perc=1.0)
    HIRS_SHR_25 = BndParmT(res=200, lim=100, score_perc=0.25)


class Boundaries(Regions):
    def __init__(self, chrm: Chromosome, regions: RegionsInternal):
        super().__init__(chrm, regions)

    def fig_subdir(self):
        return f"{FigSubDir.BOUNDARIES}/{self.chrm}_{self}"

    def nearest_rgns(self, rgns: Regions) -> Regions:
        return rgns[rgns[MIDDLE].isin(self.nearest_loc(rgns[MIDDLE]))]

    def nearest_loc(self, locs: Iterable[PosOneIdx]) -> NDArray[(Any,), PosOneIdx]:
        locs = sorted(locs)
        return np.array(
            [self._nearest_loc(getattr(bndry, MIDDLE), locs) for bndry in self]
        )

    def nearest_locs_distnc(self, locs: Iterable[PosOneIdx]) -> NDArray[(Any,), float]:
        locs = sorted(locs)
        dists = []
        for bndry in self:
            mid = getattr(bndry, MIDDLE)
            dists.append(mid - self._nearest_loc(mid, locs))

        return np.array(dists)

    def nearest_tss_distnc(self, genes: Iterable[PosOneIdx]) -> NDArray[(Any,), float]:
        ts = sorted(genes[START])
        dists = []
        for bnd in self:
            mid = getattr(bnd, MIDDLE)
            nloc = self._nearest_loc(mid, ts)
            s = genes[STRAND][genes[START] == nloc]
            dists.append(int(s) * (mid - nloc))

        return np.array(dists)

    def _nearest_loc(self, frm: PosOneIdx, locs: Iterable[PosOneIdx]) -> PosOneIdx:
        min_dst = self.chrm.total_bp
        for loc in locs:
            dst = loc - frm
            if abs(dst) > abs(min_dst):
                break
            if abs(dst) < abs(min_dst):
                min_dst = dst

        return frm + min_dst

    def quartiles(self) -> tuple[Boundaries]:
        gr = self._regions.groupby(pd.qcut(self[SCORE], 4, labels=False))
        return (self._new(gr.get_group(i)) for i in range(4))


class BoundariesHE(Boundaries):
    """
    Represenation of boundaries in a chromosome found with Hi-C Explorer

    Note: Here domains are also determined by boundaries
    """

    def __init__(
        self,
        chrm: Chromosome,
        res: PositiveInt = 500,
        lim: PositiveInt = 250,
        score_perc: ZeroToOne = 1.0,
        regions: RegionsInternal = None,
    ):
        self.res = res
        self.lim = lim
        self.score_perc = score_perc
        self._prmtr_bndrs = None
        super().__init__(chrm, regions)

    def __str__(self):
        return f"bndh_res_{self.res}_lim_{self.lim}_perc_{self.score_perc}"

    def _get_regions(
        self,
    ) -> pd.DataFrame[START:int, END:int, SCORE:float]:
        """lower score = higher separation"""
        df = pd.read_table(
            f"{PathObtain.input_dir()}/domains/"
            f"{self.chrm.number}_res_{self.res}_hicexpl_boundaries.bed",
            delim_whitespace=True,
            header=None,
            names=["chromosome", "resl", "resr", "id", SCORE, "_"],
        ).drop(columns=["chromosome", "id", "_"])

        mids = ((df["resl"] + df["resr"]) / 2).astype(int)
        df[START] = mids - self.lim
        df[END] = mids + self.lim
        df = df.drop(columns=["resl", "resr"])

        df = df.loc[df[SCORE] <= np.quantile(df[SCORE].to_numpy(), self.score_perc)]
        return df

    def _new(self, regions: RegionsInternal) -> BoundariesHE:
        return BoundariesHE(self.chrm, self.res, self.lim, self.score_perc, regions)

    def _extended(self, rgns: RegionsInternal) -> BoundariesF:
        return BoundariesHE(
            chrm=self.chrm,
            res=self.res,
            lim=int((rgns.iloc[0][END] - rgns.iloc[0][START]) / 2),
            score_perc=self.score_perc,
            regions=rgns,
        )

    def prmtr_bndrs(self) -> BoundariesHE:
        prmtrs = Promoters(self.chrm)
        return Attr.calc_attr(
            self, "_prmtr_bndrs", lambda: self.mid_contained_in(prmtrs)
        )

    def non_prmtr_bndrs(self) -> BoundariesHE:
        return self - self.prmtr_bndrs()


class DomainsHE(Regions):
    def __init__(self, bndh: BoundariesHE, regions: RegionsInternal = None):
        self._bnd = bndh
        super().__init__(bndh.chrm, regions)

    def __str__(self):
        return f"dmnsh_{self._bnd}"

    def _get_regions(self) -> pd.DataFrame[START:int, END:int]:
        return self._bnd.complement()

    def _new(self, regions: RegionsInternal) -> DomainsHE:
        return DomainsHE(bndh=self._bnd, regions=regions)


class BndFParmT(TypedDict, total=False):
    lim: int
    top_perc: float
    min_lnk: int


class BndFParm:
    SHR_25 = BndFParmT(lim=100, top_perc=0.25)
    WD_25 = BndFParmT(lim=250, top_perc=0.25)
    SHR_50 = BndFParmT(lim=100, top_perc=0.50)
    WD_50 = BndFParmT(lim=250, top_perc=0.50)
    W2D_50 = BndFParmT(lim=500, top_perc=0.50)
    SHR_50_LNK_0 = BndFParmT(lim=100, top_perc=0.50, min_lnk=0)
    SHR_50_LNK_10 = BndFParmT(lim=100, top_perc=0.50, min_lnk=10)
    SHR_50_LNK_20 = BndFParmT(lim=100, top_perc=0.50, min_lnk=20)


class BoundariesF(Boundaries):
    def __init__(
        self,
        chrm: Chromosome,
        lim: int = 100,
        top_perc: ZeroToOne = 1.0,
        regions: RegionsInternal = None,
    ):
        self._res = 200
        self.lim = lim
        self._top_perc = top_perc
        super().__init__(chrm, regions)

    def __str__(self):
        return f"bndf_res_{self._res}_lim_{self.lim}_perc_{self._top_perc}"

    def _get_regions(self):
        df = pd.read_csv(
            f"{PathObtain.gen_data_dir()}/boundaries/chrmall_res_{self._res}_w_5000_fanc.tsv",
            sep="\t",
        )
        assert START in df.columns.tolist()
        assert END in df.columns.tolist()
        assert SCORE in df.columns.tolist()

        df = df.loc[df["chromosome"] == self.chrm.number].drop(columns=["chromosome"])
        df = df.loc[df[SCORE] >= np.quantile(df[SCORE].to_numpy(), 1 - self._top_perc)]
        return df

    def _new(self, regions: RegionsInternal) -> BoundariesF:
        return BoundariesF(
            chrm=self.chrm, lim=self.lim, top_perc=self._top_perc, regions=regions
        )

    def _extended(self, rgns: RegionsInternal) -> BoundariesF:
        return BoundariesF(
            chrm=self.chrm,
            lim=int((rgns.iloc[0][END] - rgns.iloc[0][START]) / 2),
            top_perc=self._top_perc,
            regions=rgns,
        )


class DomainsF(Regions):
    def __init__(self, bndrs: BoundariesF, regions: RegionsInternal = None):
        self._bndrs = bndrs
        super().__init__(bndrs.chrm, regions)

    def __str__(self):
        return f"dmnsf_{self._bndrs}"

    def _get_regions(self) -> pd.DataFrame[START:int, END:int]:
        return self._bndrs.complement()

    def _new(self, regions: RegionsInternal) -> DomainsF:
        return DomainsF(bndrs=self._bndrs, regions=regions)


class BoundariesFN(Boundaries):
    def __init__(
        self,
        chrm: Chromosome,
        lim: int = 100,
        min_lnk: int = 10,
        top_perc: ZeroToOne = 1.0,
        regions: RegionsInternal = None,
    ):
        self._res = 200
        self._lim = lim
        self._min_lnk = min_lnk
        self._top_perc = top_perc
        self._bndrsf = BoundariesF(chrm, lim, top_perc, regions)
        super().__init__(chrm, regions)

    def __str__(self):
        return f"bfn_lnk_{self._min_lnk}_{str(self._bndrsf)}"

    def _get_regions(self):
        # Use DataCache
        ndrs = Linkers(self.chrm).ndrs(self._min_lnk)
        new_mids = self._bndrsf.nearest_loc(ndrs[MIDDLE])

        df = pd.DataFrame({SCORE: self._bndrsf[SCORE], MIDDLE: new_mids})
        df[START], df[END] = new_mids - self._lim + 1, new_mids + self._lim
        return df

    def _new(self, regions: RegionsInternal) -> BoundariesFN:
        return BoundariesFN(
            chrm=self.chrm,
            lim=self._lim,
            min_lnk=self._min_lnk,
            top_perc=self._top_perc,
            regions=regions,
        )

    def _extended(self, rgns: RegionsInternal) -> BoundariesF:
        return BoundariesFN(
            chrm=self.chrm,
            lim=(rgns.iloc[0][END] - rgns.iloc[0][START] + 1) // 2,
            min_lnk=self._min_lnk,
            top_perc=self._top_perc,
            regions=rgns,
        )

    def save_regions(self, fname: str = None) -> Path:
        return FileSave.tsv_gdatadir(
            self._regions[[START, END, SCORE]].sort_values(START),
            f"{GDataSubDir.BOUNDARIES}/{self.chrm.number}_res_200_w_5000_fancn.tsv",
        )


class DomainsFN(Regions):
    def __init__(self, bndrs: BoundariesFN, regions: RegionsInternal = None):
        self._bndrs = bndrs
        super().__init__(bndrs.chrm, regions)

    def __str__(self):
        return f"dmnsfn_{self._bndrs}"

    def _get_regions(self) -> pd.DataFrame[START:int, END:int]:
        return self._bndrs.complement()

    def _new(self, regions: RegionsInternal) -> DomainsFN:
        return DomainsFN(bndrs=self._bndrs, regions=regions)


class BoundariesType(Enum):
    HEXP = auto()
    FANC = auto()
    FANCN = auto()


@dataclass(frozen=True)
class BndSel:
    typ: BoundariesType
    parm: BndParmT | BndFParmT


class BoundariesFactory:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm

    def get_bndrs(self, bsel: BndSel) -> Boundaries:
        if bsel.typ == BoundariesType.HEXP:
            B = BoundariesHE
        elif bsel.typ == BoundariesType.FANC:
            B = BoundariesF
        elif bsel.typ == BoundariesType.FANCN:
            B = BoundariesFN

        return B(self._chrm, **(bsel.parm))


class PlotBoundariesHE:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self._bndrs = BoundariesHE(chrm, res=200, score_perc=0.5)
        self._figsubdir = FigSubDir.BOUNDARIES

    def line_c0_around_indiv(self) -> None:
        for bndry in self._bndrs.prmtr_bndrs():
            self._line_c0_around_indiv(bndry, "prmtr")

        for bndry in self._bndrs.non_prmtr_bndrs():
            self._line_c0_around_indiv(bndry, "nonprmtr")

    def _line_c0_around_indiv(self, bndry: BoundaryNT, pstr: str):
        mid = getattr(bndry, MIDDLE)
        lim = self._bndrs.lim
        self._chrm.plot_moving_avg(mid - lim, mid + lim)
        plt.xticks(ticks=[mid - lim, mid, mid + lim], labels=[-lim, 0, +lim])
        plt.xlabel(f"Distance from boundary middle")
        plt.ylabel("Intrinsic Cyclizability")
        plt.title(
            f"C0 around {pstr} boundary at {mid} bp of chromosome {self._chrm.number}."
            f"Found with res {self._bndrs.res}"
        )

        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/{self._chrm.id}/{pstr}_indiv_{mid}.png"
        )

    def scatter_mean_c0_at_indiv(self) -> Path:
        markers = ["o", "s"]
        labels = ["promoter", "non-promoter"]
        colors = ["tab:blue", "tab:orange"]

        PlotUtil.clearfig()
        PlotUtil.show_grid()

        p_x = self._bndrs.prmtr_bndrs()[MIDDLE]
        np_x = self._bndrs.non_prmtr_bndrs()[MIDDLE]
        plt.scatter(
            p_x,
            self._bndrs.prmtr_bndrs()[MEAN_C0],
            marker=markers[0],
            label=labels[0],
            color=colors[0],
        )
        plt.scatter(
            np_x,
            self._bndrs.non_prmtr_bndrs()[MEAN_C0],
            marker=markers[1],
            label=labels[1],
            color=colors[1],
        )

        horiz_colors = ["tab:green", "tab:red", "tab:purple"]
        chrm_mean_c0 = self._chrm.c0_spread().mean()
        PlotUtil.horizline(DomainsHE(self._chrm).mean_c0, horiz_colors[0], "domains")
        PlotUtil.horizline(chrm_mean_c0, horiz_colors[1], "chromosome")
        PlotUtil.horizline(self._bndrs.mean_c0, horiz_colors[2], "boundaries")

        plt.xlabel("Position along chromosome (bp)")
        plt.ylabel("Mean C0")
        plt.title(
            f"Comparison of mean {self._chrm.c0_type} C0 among boundaries"
            f" in chromosome {self._chrm.number}"
        )
        plt.legend()

        return FileSave.figure_in_figdir(
            f"{self._figsubdir}/mean_c0_scatter_{self._bndrs}.png"
        )


class BoundariesDomainsHEQuery:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._boundaries = BoundariesHE(chrm)
        self._domains = DomainsHE(chrm)

    def prmtr_bndrs_mean_c0(self) -> float:
        return self._chrm.mean_c0_of_segments(
            self._boundaries.prmtr_bndrs()[MIDDLE],
            self._boundaries._lim,
            self._boundaries._lim,
        )

    def non_prmtr_bndrs_mean_c0(self) -> float:
        return self._chrm.mean_c0_of_segments(
            self._boundaries.non_prmtr_bndrs()[MIDDLE],
            self._boundaries._lim,
            self._boundaries._lim,
        )

    def num_bndry_mean_c0_greater_than_dmn(self) -> PositiveInt:
        return (self._boundaries[MEAN_C0] > self._domains.mean_c0).sum()

    def num_prmtr_bndry_mean_c0_greater_than_dmn(self) -> float:
        return (self._boundaries.prmtr_bndrs()[MEAN_C0] > self._domains.mean_c0).sum()

    def num_non_prmtr_bndry_mean_c0_greater_than_dmns(self) -> float:
        return (
            self._boundaries.non_prmtr_bndrs()[MEAN_C0] > self._domains.mean_c0
        ).sum()


class MCBoundariesHE:
    def __init__(self, mchrm: MultiChrm):
        self._mchrm = mchrm
        self._bndrs = list(map(lambda c: BoundariesHE(c), mchrm))

    def __iter__(self):
        return iter(self._bndrs)

    def __str__(self):
        return str(self._mchrm)


class MCDomainsHE:
    def __init__(self, mchrm: MultiChrm):
        self._mchrm = mchrm
        self._dmns = list(map(lambda c: DomainsHE(c), mchrm))

    def __iter__(self):
        return iter(self._dmns)

    def __str__(self):
        return str(self._mchrm)


# TODO:
# Rewrite by using MCBoundaries and removing unnecessary procedures
class MCBoundariesHECollector:
    def __init__(
        self,
        prediction: Prediction,
        chrids: tuple[ChrId] = ChrIdList,
        res: PositiveInt = 500,
        lim: PositiveInt = 250,
    ):
        self._prediction = prediction
        self._chrids = chrids
        self._res = res
        self._lim = lim
        self._coll_df = pd.DataFrame({"ChrID": chrids})
        self._chrms = self._get_chrms()
        self._mc_bndrs = self._get_mc_bndrs()

    def __str__(self):
        ext = "with_vl" if "VL" in self._chrids else "without_vl"
        return f"res_{self._res}_lim_{self._lim}_md_{self._prediction}_{ext}"

    def _get_chrms(self) -> pd.Series:
        return pd.Series(
            list(
                map(lambda chrm_id: Chromosome(chrm_id, self._prediction), self._chrids)
            )
        )

    def _get_mc_bndrs(self) -> pd.Series[BoundariesHE]:
        return self._chrms.apply(lambda chrm: BoundariesHE(chrm, self._res, self._lim))

    def _add_num_bndrs(self) -> Literal["num_bndrs"]:
        num_bndrs_col = "num_bndrs"
        if num_bndrs_col not in self._coll_df.columns:
            self._coll_df[num_bndrs_col] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df)
            )

        return num_bndrs_col

    def _add_num_dmns(self) -> str:
        num_dmns_col = "num_dmns"
        if num_dmns_col not in self._coll_df.columns:
            num_bndrs_col = self._add_num_bndrs()
            self._coll_df[num_dmns_col] = self._coll_df[num_bndrs_col] - 1

        return num_dmns_col

    def _add_bndrs_mean(self) -> Literal["c0_bndrs"]:
        c0_bndrs_col = "c0_bndrs"
        if c0_bndrs_col not in self._coll_df.columns:
            self._coll_df["c0_bndrs"] = self._mc_bndrs.apply(
                lambda bndrs: bndrs.bndry_domain_mean_c0()[0]
            )

        return c0_bndrs_col

    def _add_dmns_mean(self) -> Literal["c0_dmns"]:
        c0_dmns_col = "c0_dmns"
        if c0_dmns_col not in self._coll_df.columns:
            self._coll_df[c0_dmns_col] = self._mc_bndrs.apply(
                lambda bndrs: bndrs.bndry_domain_mean_c0()[1]
            )

        return c0_dmns_col

    def _add_num_bndrs_gt_dmns(self) -> str:
        num_bndrs_gt_dmns_col = "num_bndrs_gt_dmns"
        if num_bndrs_gt_dmns_col not in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[num_bndrs_gt_dmns_col] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return num_bndrs_gt_dmns_col

    def _add_num_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 5
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df.query("in_promoter")["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return self.col_for(method_id)

    def _add_num_non_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 6
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name="mc_bndrs")
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (
                    bndrs["mc_bndrs"].bndrs_df.query("not in_promoter")["mean_c0"]
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]
                ).sum(),
                axis=1,
            )

        return self.col_for(method_id)

    def _add_num_prmtr_bndrs(self) -> Literal["num_p_b"]:
        method_id = 7
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query("in_promoter"))
            )

        return self.col_for(method_id)

    def _add_num_non_prmtr_bndrs(self) -> Literal["num_np_b"]:
        method_id = 8
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query("not in_promoter"))
            )

        return self.col_for(method_id)

    def col_for(self, method_id: int) -> str:
        col_map = {
            0: "num_bndrs",
            1: "num_dmns",
            2: "c0_bndrs",
            3: "c0_dmns",
            4: "num_bndrs_gt_dmns",
            5: "num_p_b_gt_d",
            6: "num_np_b_gt_d",
            7: "num_p_b",
            8: "num_np_b",
        }
        return col_map[method_id]

    def save_stat(self, methods: list[int] = None) -> Path:
        method_map = {
            0: self._add_num_bndrs,
            1: self._add_num_dmns,
            2: self._add_bndrs_mean,
            3: self._add_dmns_mean,
            4: self._add_num_bndrs_gt_dmns,
            5: self._add_num_prmtr_bndrs_gt_dmns,
            6: self._add_num_non_prmtr_bndrs_gt_dmns,
            7: self._add_num_prmtr_bndrs,
            8: self._add_num_non_prmtr_bndrs,
        }
        for m in methods:
            method_map[m]()

        self._coll_df["res"] = np.full((len(self._coll_df),), self._res)
        self._coll_df["lim"] = np.full((len(self._coll_df),), self._lim)
        self._coll_df["model"] = np.full((len(self._coll_df),), str(self._prediction))

        return FileSave.append_tsv(
            self._coll_df,
            f"{PathObtain.data_dir()}/generated_data/mcdomains/mcdmns_stat.tsv",
        )

    def plot_scatter_mean_c0(self) -> Path:
        """Draw scatter plot of mean c0 at boundaries and domains of
        chromosomes"""
        chrms = self._chrms
        chrm_means = chrms.apply(lambda chrm: chrm.c0_spread().mean())

        mc_bndrs = self._mc_bndrs
        mc_prmtr_bndrs_c0 = mc_bndrs.apply(lambda bndrs: bndrs.prmtr_bndrs_mean_c0())
        mc_non_prmtr_bndrs_c0 = mc_bndrs.apply(
            lambda bndrs: bndrs.non_prmtr_bndrs_mean_c0()
        )

        mc_bndrs_dmns_c0 = mc_bndrs.apply(lambda bndrs: bndrs.bndry_domain_mean_c0())
        mc_bndrs_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:, 0]
        mc_dmns_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:, 1]

        # Print comparison
        print("bndrs > dmns:", (mc_bndrs_c0 > mc_dmns_c0).sum())
        print("prmtr bndrs > dmns:", (mc_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print("non prmtr bndrs > dmns:", (mc_non_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print("chrms > dmns:", (chrm_means.to_numpy() > mc_dmns_c0).sum())

        PlotUtil.show_grid()
        x = np.arange(len(self._chrids))
        markers = ["o", "s", "p", "P", "*"]
        labels = [
            "chromosome",
            "promoter bndrs",
            "non-promoter bndrs",
            "boundaries",
            "domains",
        ]

        for i, y in enumerate(
            (
                chrm_means,
                mc_prmtr_bndrs_c0,
                mc_non_prmtr_bndrs_c0,
                mc_bndrs_c0,
                mc_dmns_c0,
            )
        ):
            plt.scatter(x, y, marker=markers[i], label=labels[i])

        plt.xticks(x, self._chrids)
        plt.xlabel("Chromosome")
        plt.ylabel("Mean C0")
        plt.title(f"Comparison of mean C0 in boundaries vs. domains")
        plt.legend()

        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcdomains/bndrs_dmns_c0_{self}.png"
        )

    def plot_bar_perc_in_prmtrs(self) -> Path:
        chrms = self._chrms
        mc_bndrs = chrms.apply(lambda chrm: BoundariesHE(chrm, self._res))
        perc_in_prmtrs = mc_bndrs.apply(
            lambda bndrs: Promoters(bndrs._chrm)
            .is_in_regions(bndrs.bndrs_df["middle"])
            .mean()
            * 100
        )

        PlotUtil.show_grid()
        x = np.arange(len(self._chrids))
        plt.bar(x, perc_in_prmtrs)
        plt.xticks(x, self._chrids)
        plt.xlabel("Chromosome")
        plt.ylabel("Boundaries in promoters (%)")
        plt.title(f"Percentage of boundaries in promoters in chromosomes")
        plt.legend()
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcdomains/perc_bndrs_in_promoters_{self}.png"
        )

    def num_bndrs_dmns(self) -> tuple[float, float]:
        mc_bndrs = self._mc_bndrs
        num_bndrs = mc_bndrs.apply(lambda bndrs: len(bndrs.bndrs_df)).sum()
        num_dmns = num_bndrs - len(self._chrids)
        return num_bndrs, num_dmns

    def mean_dmn_len(self) -> float:
        mc_bndrs = self._mc_bndrs
        return (
            mc_bndrs.apply(lambda bndrs: bndrs.get_domains()["len"].sum()).sum()
            / self.num_bndrs_dmns()[1]
        )

    def individual_bndry_stat(self) -> None:
        # TODO: Reduce function calls. Access index with .name if needed.
        mc_bndrs = self._mc_bndrs
        num_mc_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_bndry_mean_c0_greater_than_dmn(bndrs)
        ).sum()
        print("num_mc_bndrs_gt", num_mc_bndrs_gt)
        num_mc_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_prmtr_bndry_mean_c0_greater_than_dmn(bndrs)
        ).sum()
        print("num_mc_prmtr_bndrs_gt", num_mc_prmtr_bndrs_gt)

        num_mc_non_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: BoundariesHE.num_non_prmtr_bndry_mean_c0_greater_than_dmns(
                bndrs
            )
        ).sum()
        print("num_mc_non_prmtr_bndrs_gt", num_mc_non_prmtr_bndrs_gt)

        num_mc_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query("in_promoter"))
        ).sum()
        print("num_mc_prmtr_bndrs", num_mc_prmtr_bndrs)

        num_mc_non_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query("not in_promoter"))
        ).sum()
        print("num_mc_non_prmtr_bndrs", num_mc_non_prmtr_bndrs)


class MCBoundariesHEAggregator:
    def __init__(self, coll: MCBoundariesHECollector):
        self._coll = coll
        self._agg_df = pd.DataFrame({"ChrIDs": [coll._coll_df["ChrID"].tolist()]})

    def _bndrs_gt_dmns(self):
        self._coll.save_stat([0, 4])
        self._agg_df["b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(4)].sum()
            / self._coll._coll_df[self._coll.col_for(0)].sum()
            * 100
        )

    def _p_bndrs_gt_dmns(self):
        self._coll.save_stat([7, 5])
        self._agg_df["p_b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(5)].sum()
            / self._coll._coll_df[self._coll.col_for(7)].sum()
            * 100
        )

    def _np_bndrs_gt_dmns(self):
        self._coll.save_stat([8, 6])
        self._agg_df["np_b_gt_d"] = (
            self._coll._coll_df[self._coll.col_for(6)].sum()
            / self._coll._coll_df[self._coll.col_for(8)].sum()
            * 100
        )

    def _num_bndrs(self):
        self._coll.save_stat([0, 7, 8])
        self._agg_df["num_b"] = self._coll._coll_df[self._coll.col_for(0)].sum()
        self._agg_df["num_p_b"] = self._coll._coll_df[self._coll.col_for(7)].sum()
        self._agg_df["num_np_b"] = self._coll._coll_df[self._coll.col_for(8)].sum()

    def save_stat(self):
        self._num_bndrs()
        self._bndrs_gt_dmns()
        self._p_bndrs_gt_dmns()
        self._np_bndrs_gt_dmns()
        self._agg_df["res"] = np.full((len(self._agg_df),), self._coll._res)
        self._agg_df["lim"] = np.full((len(self._agg_df),), self._coll._lim)
        self._agg_df["model"] = np.full(
            (len(self._agg_df),), str(self._coll._prediction)
        )

        return FileSave.append_tsv(
            self._agg_df,
            f"{PathObtain.data_dir()}/generated_data/mcdomains/aggr_mcdmns_stat.tsv",
        )
