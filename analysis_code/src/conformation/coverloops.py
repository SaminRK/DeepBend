from __future__ import annotations
from typing import Callable, Iterable, NamedTuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from nptyping import NDArray

from conformation.loops import MCLoops

from .loops import Loops, COL_START, COL_END, COL_MEAN_C0_FULL
from chromosome.chromosome import Chromosome, MultiChrm
from chromosome.nucleosomes import Nucleosomes
from util.constants import ONE_INDEX_START
from util.util import DataCache, NumpyTool, PathObtain, FileSave, PlotUtil
from util.custom_types import ChrId
from util.constants import ChrIdList

# TODO: Create super class for cover classes


class CoverLoops:
    "Loop seqs determined by coverage by original loops in a chromosome"

    def __init__(self, loops: Loops):
        self._chrm = loops.chrm
        self.covermask = loops.covermask()
        self._cloops = self._coverloops_with_c0()

    def __len__(self):
        return len(self._cloops)

    def __iter__(
        self,
    ) -> Iterable[NamedTuple[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]]:
        return self._cloops.itertuples()

    def __getitem__(self, key: str) -> pd.Series:
        if key in self._cloops.columns:
            return self._cloops[key]

        raise KeyError

    @property
    def mean_c0(self) -> float:
        if not hasattr(self, "_mean_c0"):
            self._mean_c0 = self._chrm.c0_spread()[self.covermask].mean()

        return self._mean_c0

    def _coverloops_with_c0(
        self,
    ) -> pd.DataFrame[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]:
        def _calc_mean_c0() -> pd.DataFrame:
            cloops = self._coverloops()
            cloops[COL_MEAN_C0_FULL] = cloops.apply(
                lambda cl: self._chrm.mean_c0_segment(*cl[[COL_START, COL_END]]), axis=1
            )
            return cloops

        return DataCache.calc_df_tsv(
            f"loops/cover_c0_{self._chrm.id}.tsv",
            _calc_mean_c0,
        )

    def _coverloops(self) -> pd.DataFrame[COL_START:int, COL_END:int]:
        clstarts = (
            NumpyTool.match_pattern(self.covermask, [False, True]) + 1 + ONE_INDEX_START
        )
        clends = (
            NumpyTool.match_pattern(self.covermask, [True, False]) + ONE_INDEX_START
        )
        assert len(clstarts) == len(clends)

        return pd.DataFrame({COL_START: clstarts, COL_END: clends})


class NonCoverLoops:
    def __init__(self, loops: Loops):
        self._chrm = loops.chrm

        cloops = CoverLoops(loops)
        self.covermask = ~cloops.covermask
        self._ncloops = self._noncoverloops_with_c0(cloops)

    def __len__(self):
        return len(self._ncloops)

    def __iter__(
        self,
    ) -> Iterable[NamedTuple[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]]:
        return self._ncloops.itertuples()

    def __getitem__(self, key) -> pd.Series:
        if key in self._ncloops.columns:
            return self._ncloops[key]

        raise KeyError

    @property
    def mean_c0(self) -> float:
        if not hasattr(self, "_mean_c0"):
            self._mean_c0 = self._chrm.c0_spread()[self.covermask].mean()

        return self._mean_c0

    def _noncoverloops_with_c0(
        self, cloops: CoverLoops
    ) -> pd.DataFrame[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]:
        def _calc_mean_c0() -> pd.DataFrame:
            ncloops = self._noncoverloops(cloops)
            ncloops[COL_MEAN_C0_FULL] = ncloops.apply(
                lambda ncl: self._chrm.mean_c0_segment(*ncl[[COL_START, COL_END]]),
                axis=1,
            )
            return ncloops

        return DataCache.calc_df_tsv(
            f"loops/noncover_c0_{self._chrm.id}.tsv",
            _calc_mean_c0,
        )

    def _noncoverloops(
        self, cloops: CoverLoops
    ) -> pd.DataFrame[COL_START:int, COL_END:int]:
        nlstarts = np.append([ONE_INDEX_START], cloops[COL_END] + 1)
        nlends = np.append(cloops[COL_START] - 1, self._chrm.total_bp)
        return pd.DataFrame({COL_START: nlstarts, COL_END: nlends})


class PlotCoverLoops:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._loops = Loops(self._chrm)

    def plot_histogram_c0(self) -> Path:
        PlotUtil.clearfig()
        plt.hist(CoverLoops(self._loops)[COL_MEAN_C0_FULL], label="Loops", alpha=0.5)
        plt.hist(
            NonCoverLoops(self._loops)[COL_MEAN_C0_FULL], label="Non-loops", alpha=0.5
        )
        plt.legend()
        return FileSave.figure_in_figdir(f"loops/hist_c0_{self._chrm}.png")


class MCCoverLoops:
    def __init__(self, mcloops: MCLoops):
        self._mccloops = self._mccoverloops_with_c0(mcloops)

    def __len__(self):
        return len(self._mccloops)

    def __iter__(
        self,
    ) -> Iterable[NamedTuple[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]]:
        return self._mccloops.itertuples()

    def __getitem__(self, key) -> pd.Series:
        if key in self._mccloops.columns:
            return self._mccloops[key]

        raise KeyError

    def _mccoverloops_with_c0(
        self, mcloops: MCLoops
    ) -> pd.DataFrame[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]:
        mccloops = list(map(lambda loops: CoverLoops(loops), mcloops))
        return pd.DataFrame([cl for clps in mccloops for cl in clps])


class MCNonCoverLoops:
    def __init__(self, mcloops: MCLoops):
        self._mcncloops = self._mcncloops_with_c0(mcloops)

    def __len__(self):
        return len(self._mcncloops)

    def __iter__(
        self,
    ) -> Iterable[NamedTuple[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]]:
        return self._mcncloops.itertuples()

    def __getitem__(self, key) -> pd.Series:
        if key in self._mcncloops.columns:
            return self._mcncloops[key]

        raise KeyError

    def _mcncloops_with_c0(
        self, mcloops: MCLoops
    ) -> pd.DataFrame[COL_START:int, COL_END:int, COL_MEAN_C0_FULL:float]:
        mcncloops = list(map(lambda loops: NonCoverLoops(loops), mcloops))
        return pd.DataFrame([ncl for nclps in mcncloops for ncl in nclps])


class PlotMCCoverLoops:
    def __init__(self, mcchrm: MultiChrm):
        self._mcloops = MCLoops(mcchrm)

    def box_plot_c0(self) -> Path:
        PlotUtil.show_grid(which="both")
        showfliers = False
        plt.boxplot(
            [
                MCCoverLoops(self._mcloops)[COL_MEAN_C0_FULL],
                MCNonCoverLoops(self._mcloops)[COL_MEAN_C0_FULL],
            ],
            showfliers=showfliers,
        )
        plt.xticks(ticks=[1, 2], labels=["Loops", "Non-loops"])
        plt.ylabel("Mean C0")
        plt.title(
            "Comparison of mean c0 distribution of loops and non-loops in all chromosomes"
        )
        return FileSave.figure_in_figdir(
            f"mcloops/box_plot_fl_{showfliers}_{self._mcloops}.png"
        )

    def plot_histogram_c0(self):
        num_bins = 40
        bins = np.linspace(-0.4, 0.0, num_bins)
        plt.hist(
            MCCoverLoops(self._mcloops)[COL_MEAN_C0_FULL],
            bins,
            label="Loops",
            alpha=0.8,
        )
        plt.hist(
            MCNonCoverLoops(self._mcloops)[COL_MEAN_C0_FULL],
            bins,
            label="Non-loops",
            alpha=0.5,
        )
        plt.legend()
        plt.xlabel("Mean C0")
        plt.ylabel("Number of items")
        plt.title(
            "Comparison of distribution of mean c0 of loops and non-loops in all chromosomes"
        )
        return FileSave.figure_in_figdir(
            f"mcloops/hist_c0_bins_{num_bins}_{self._mcloops}.png"
        )


class LoopsCover:
    def __init__(self, loops: Loops):
        nucs = Nucleosomes(loops._chr)

        self._nuc_cover = nucs.get_nuc_regions()
        self._loop_cover = loops.covermask(loops._loop_df)

    def in_loop_nuc(self) -> float:
        return (self._loop_cover & self._nuc_cover).mean()

    def in_loop_linker(self) -> float:
        return (self._loop_cover & ~self._nuc_cover).mean()

    def in_non_loop_nuc(self) -> float:
        return (~self._loop_cover & self._nuc_cover).mean()

    def in_non_loop_linker(self) -> float:
        return (~self._loop_cover & ~self._nuc_cover).mean()


class MultiChrmLoopsCoverCollector:
    # TODO: Use MultiChrm and MCLoops
    def __init__(self, chrmids: tuple[ChrId] = ChrIdList, mxlen: int | None = None):
        self._chrmids = chrmids
        self._mxlen = mxlen

        chrms = pd.Series(list(map(lambda chrm_id: Chromosome(chrm_id), chrmids)))
        mcloops = chrms.apply(lambda chrm: Loops(chrm, mxlen))
        self._mccloops = mcloops.apply(lambda loops: LoopsCover(loops))

    def get_cover_stat(self) -> pd.DataFrame:
        collector_df = pd.DataFrame({"ChrID": self._chrmids})

        collector_df["loop_nuc"] = self._mccloops.apply(
            lambda cloops: cloops.in_loop_nuc()
        )

        collector_df["loop_linker"] = self._mccloops.apply(
            lambda cloops: cloops.in_loop_linker()
        )

        collector_df["non_loop_nuc"] = self._mccloops.apply(
            lambda cloops: cloops.in_non_loop_nuc()
        )

        collector_df["non_loop_linker"] = self._mccloops.apply(
            lambda cloops: cloops.in_non_loop_linker()
        )
        save_path_str = f"{PathObtain.data_dir()}/generated_data/mcloops/multichr_cover_stat_{self._mxlen}.tsv"
        FileSave.append_tsv(collector_df, save_path_str)
        return collector_df, save_path_str

    def plot_bar_cover_stat(self) -> str:
        labels = ["loop_nuc", "loop_linker", "non_loop_nuc", "non_loop_linker"]
        colt_df = self.get_cover_stat()[0]
        colt_arr = colt_df[labels].values
        mpl.rcParams.update({"font.size": 12})
        PlotUtil.bar_stacked(
            colt_arr.transpose() * 100,
            labels,
            colt_df["ChrID"].tolist(),
            show_values=True,
            value_format="{:.1f}",
            y_label="Coverage (%)",
        )

        plt.gca().legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=4,
            fancybox=False,
            shadow=False,
        )

        plt.xlabel("Chromosome")
        plt.title(
            "Coverage by nucleosomes and linkers in loop and"
            f"non-loop region with max loop length = {self._mxlen}",
            pad=35,
        )
        fig_path_str = (
            f"{PathObtain.figure_dir()}/mcloops/nuc_linker_cover_mxl_{self._mxlen}.png"
        )

        FileSave.figure(fig_path_str)
        return fig_path_str
