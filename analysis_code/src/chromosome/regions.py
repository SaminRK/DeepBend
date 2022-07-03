from __future__ import annotations
from pathlib import Path
from typing import Iterable, NamedTuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray


from chromosome.chromosome import ChrmOperator, Chromosome, PlotChrm
from util.util import Attr, FileSave, NumpyTool, PlotUtil
from util.constants import ONE_INDEX_START, MAX_TOTAL_BP
from util.custom_types import PosOneIdx, NonNegativeInt, PositiveInt, C0

RegionsInternal = pd.DataFrame

START = "start"
END = "end"
LEN = "len"
MIDDLE = "middle"
MEAN_C0 = "mean_c0"
C0_QUARTILE = "c0_quartile"


class RegionNT(NamedTuple):
    start: PosOneIdx
    end: PosOneIdx
    len: int
    middle: PosOneIdx
    mean_c0: float
    c0_quartile: list[float]


class Regions:
    def __init__(self, chrm: Chromosome, regions: RegionsInternal = None) -> None:
        self.chrm = chrm
        self._regions = (
            regions.reset_index(drop=True)
            if regions is not None
            else self._get_regions().reset_index(drop=True)
        )
        self._add_len()
        self._add_middle()
        self._add_mean_c0()
        self._add_c0_quartile()
        self._mean_c0 = None
        self._cvrmask = None

    gdata_savedir = None

    def __iter__(self) -> Iterable[RegionNT]:
        return self._regions.itertuples()

    def __getitem__(self, key: NonNegativeInt | str | Iterable) -> pd.Series | Regions:
        if isinstance(key, NonNegativeInt):
            return self._regions.iloc[key]

        # Probably not needed
        # def _is_bool_array(arg):
        #     return isinstance(arg, Iterable) and isinstance(list(arg)[0], bool)

        # if _is_bool_array(key):
        #     return self._new(self._regions.loc[key])

        if isinstance(key, str):
            if key in self._regions.columns:
                return self._regions[key]
            raise KeyError

        if isinstance(key, Iterable):
            return self._new(self._regions[key])

        raise KeyError

    def __len__(self):
        return len(self._regions)

    def __sub__(self, other: Regions) -> Regions:
        result_rgns = self._regions[
            ~(
                self._regions[START].isin(other._regions[START])
                & self._regions[END].isin(other._regions[END])
            )
        ]
        return self._new(result_rgns)

    def __add__(self, other: Regions) -> Regions:
        assert set(self._regions.columns) == set(other._regions.columns)
        return self._new(pd.merge(self._regions, other._regions, how="outer"))

    def __str__(self):
        return "regions"

    @property
    def mean_c0(self) -> float:
        def calc_mean_c0():
            return self.chrm.c0_spread()[self.cover_mask].mean()

        return Attr.calc_attr(self, "_mean_c0", calc_mean_c0)

    @property
    def total_bp(self) -> int:
        return self.cover_mask.sum()

    @property
    def cover_mask(self) -> NDArray[(Any,), bool]:
        def calc_cvrmask():
            return ChrmOperator(self.chrm).create_cover_mask(
                self._regions[START], self._regions[END]
            )

        return Attr.calc_attr(self, "_cvrmask", calc_cvrmask)

    def c0(self) -> NDArray[(Any, Any), C0]:
        return ChrmOperator(self.chrm).c0_rgns(self[START], self[END])

    def seq(self) -> list[str]:
        return [self.chrm.seq[s - 1 : e] for s, e in zip(self[START], self[END])]

    def complement(self) -> RegionsInternal:
        if self._uniform_len():
            rgns = self
        else:
            rgns = self.cover_regions()

        rgns_df = rgns._regions.sort_values(by=START)
        df = pd.DataFrame(
            {
                START: np.concatenate(
                    [
                        [
                            ONE_INDEX_START,
                        ],
                        rgns_df[END] + 1,
                    ]
                ),
                END: np.concatenate(
                    [
                        rgns_df[START] - 1,
                        [
                            self.chrm.total_bp,
                        ],
                    ]
                ),
            }
        )
        return df.loc[df[START] <= df[END]]

    def sort(self, by: str) -> Regions:
        return self._new(
            self._regions.sort_values(by=by, inplace=False, ignore_index=True)
        )

    def is_in_regions(self, bps: Iterable[PosOneIdx]) -> NDArray[(Any,), bool]:
        return np.array([self.cover_mask[bp - 1] for bp in bps])

    def cover_regions(self) -> Regions:
        cvr = np.concatenate(([False], self.cover_mask, [False]))
        starts = NumpyTool.match_pattern(cvr, [False, True]) + ONE_INDEX_START
        ends = NumpyTool.match_pattern(cvr, [True, False])
        assert len(starts) == len(ends)

        rgns = Regions(self.chrm, pd.DataFrame({START: starts, END: ends}))
        return rgns

    def overlaps_with_rgns(self, rgns: Regions, min_ovbp: PositiveInt) -> Regions:
        def _overlaps(srgn: RegionNT) -> bool:
            scv = ChrmOperator(self.chrm).create_cover_mask(
                [getattr(srgn, START)], [getattr(srgn, END)]
            )
            return np.sum(scv & rgns.cover_mask) >= min_ovbp

        ovlps = list(map(lambda srgn: _overlaps(srgn), self))
        return self._new(self._regions.iloc[ovlps])

    def with_rgn(self, rgns: Regions) -> Regions:
        cntns = self._contains_rgn(rgns)
        return self._new(self._regions.iloc[cntns])

    def _contains_rgn(self, rgns: Regions) -> NDArray[(Any,), bool]:
        def _rgn_contains_rgn(cntn: RegionNT) -> bool:
            cntns = False
            for rgn in rgns:
                if getattr(cntn, START) <= getattr(rgn, START) and getattr(
                    rgn, END
                ) <= getattr(cntn, END):
                    cntns = True
                    break
            return cntns

        return np.array(list(map(lambda cntn: _rgn_contains_rgn(cntn), self)))

    def with_loc(self, locs: Iterable[PosOneIdx], with_x: bool) -> Regions:
        # TODO: Remove with_x. User can get without by subtract.
        cntns = self._contains_loc(locs)
        return self._new(self._regions.iloc[cntns if with_x else ~cntns])

    def _contains_loc(self, locs: Iterable[PosOneIdx]) -> NDArray[(Any,), bool]:
        locs = sorted(locs)

        def _rgn_contains_loc(region: RegionNT) -> bool:
            cntns = False
            for loc in locs:
                if getattr(region, START) <= loc <= getattr(region, END):
                    cntns = True
                    break
                if loc > getattr(region, END):
                    break

            return cntns

        return np.array(list(map(lambda rgn: _rgn_contains_loc(rgn), self)))

    def rgns_contained_in(self, containers: Regions) -> Regions:
        cntnd = self._rgns_contained_in(containers)
        return self._new(self._regions.iloc[cntnd])

    def _rgns_contained_in(self, containers: Regions) -> NDArray[(Any,), bool]:
        def _rgn_in_containers(rgn: RegionNT) -> bool:
            inc = False
            for cntn in containers:
                if getattr(cntn, START) <= getattr(rgn, START) and getattr(
                    rgn, END
                ) <= getattr(cntn, END):
                    inc = True
                    break

            return inc

        return np.array(list(map(lambda rgn: _rgn_in_containers(rgn), self)))

    def mid_contained_in(self, containers: Regions) -> Regions:
        cntnrs = containers.sort(START)

        def _alg1() -> Regions:
            rgns = self.sort(MIDDLE)
            mids = rgns[MIDDLE]
            cntnd = []
            ci = 0
            for mid in mids:
                while ci < len(cntnrs):
                    cntn = cntnrs[ci]
                    if (
                        getattr(cntn, START)
                        <= mid
                        <= getattr(cntn, END)
                    ):
                        cntnd.append(True)
                        break
                    if mid < getattr(cntn, START):
                        cntnd.append(False)
                        break
                    ci += 1
            
            cntnd += [False] * (len(rgns) - len(cntnd))
            return rgns[cntnd]

        def _alg2() -> Regions:
            def _mid_in_containers(mid: PosOneIdx) -> bool:
                inc = False
                for cntn in cntnrs:
                    if getattr(cntn, START) <= mid <= getattr(cntn, END):
                        inc = True
                        break

                    if mid < getattr(cntn, START):
                        break

                return inc

            cntnd = [_mid_in_containers(getattr(rgn, MIDDLE)) for rgn in self]
            return self[cntnd]

        return _alg1()

    def extended(self, ebp: int) -> Regions:
        return self._extended(
            RegionsInternal(
                {START: self._regions[START] - ebp, END: self._regions[END] + ebp}
            )
        )

    def len_in(self, mn: int = 0, mx: int = MAX_TOTAL_BP) -> Regions:
        return self._new(self._regions.query(f"{mn} <= {LEN} <= {mx}"))

    def sections(self, ln: int) -> Regions:
        rgns = self.sort(START)
        secs = np.empty((0,))
        sece = np.empty((0,))
        for rgn in rgns:
            s = np.array(
                range(
                    getattr(rgn, START) + (getattr(rgn, LEN) % ln) // 2,
                    getattr(rgn, END) - ln + 2,
                    ln,
                )
            )
            secs = np.append(secs, s)
            sece = np.append(sece, s + ln - 1)

        return Regions(
            self.chrm, pd.DataFrame({START: secs.astype(int), END: sece.astype(int)})
        )

    def save_regions(self, fname: str = None) -> Path:
        return FileSave.tsv_gdatadir(
            self._regions.sort_values(START),
            f"{type(self).__name__.lower()}/{fname or type(self).__name__.lower()}.tsv",
        )

    def _get_regions(self) -> RegionsInternal:
        """Must be implemented in subclass"""

    def _new(self, rgns: RegionsInternal) -> Regions:
        "Should be overridden in subclass if constructor does not conform to this"
        return type(self)(self.chrm, rgns.copy())

    def _extended(self, rgns: RegionsInternal) -> Regions:
        return self._new(rgns)

    def _add_len(self) -> None:
        self._regions.loc[:, LEN] = self._regions[END] - self._regions[START] + 1

    def _add_middle(self) -> None:
        self._regions.loc[:, MIDDLE] = (
            (self._regions[START] + self._regions[END]) / 2
        ).astype(int)

    def _add_mean_c0(self) -> None:
        self._regions = self._regions.assign(
            mean_c0=lambda rgns: ChrmOperator(self.chrm).mean_c0_regions_indiv(
                rgns[START], rgns[END]
            )
        )

    def _add_c0_quartile(self) -> None:
        if len(self._regions) > 0:
            self._regions.loc[:, C0_QUARTILE] = self._regions.apply(
                lambda rgn: np.round(
                    np.quantile(
                        ChrmOperator(self.chrm).c0(rgn[START], rgn[END]),
                        [0, 0.25, 0.5, 0.75, 1],
                    ),
                    3,
                ),
                axis=1,
            )
            self._regions[C0_QUARTILE] = self._regions[C0_QUARTILE].apply(tuple)

    def _uniform_len(self) -> bool:
        return len(pd.unique(self[LEN])) == 1


class PlotRegions:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0_indiv(self, rgn: RegionNT) -> None:
        PlotUtil.clearfig()
        PlotUtil.show_grid(which="minor")
        pltchrm = PlotChrm(self._chrm)
        pltchrm.line_c0(getattr(rgn, START), getattr(rgn, END))
        plt.xlabel("Position (bp)")
        plt.ylabel("C0")
