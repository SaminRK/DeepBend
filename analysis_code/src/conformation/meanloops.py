from __future__ import annotations
from pathlib import Path
import itertools
from typing import Literal, Callable
import functools
import operator

import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
import numpy as np

from chromosome.nucleosomes import Nucleosomes
from conformation.coverloops import CoverLoops, NonCoverLoops
from util.custom_types import ChrId
from models.prediction import Prediction
from util.constants import ChrIdList
from chromosome.chromosome import Chromosome
from util.util import FileSave, PlotUtil, PathObtain
from conformation.loops import Loops, COL_START, COL_END, COL_MEAN_C0_FULL, COL_LEN


class MeanLoops:
    """Class to find mean across loops in various ways"""

    def __init__(self, loops: Loops):
        self._loops = loops

    def _get_quartile_dfs(self, loops: Loops) -> tuple[Loops]:
        """Split a dataframe into 4 representing quartile on column len"""
        quart1, quart2, quart3 = loops["len"].quantile([0.25, 0.5, 0.75]).tolist()
        return (
            loops[list(filter(lambda i: loops[i]["len"] <= quart1, loops.index))],
            loops[
                list(filter(lambda i: quart1 < loops[i]["len"] <= quart2, loops.index))
            ],
            loops[
                list(filter(lambda i: quart2 < loops[i]["len"] <= quart3, loops.index))
            ],
            loops[list(filter(lambda i: quart3 < loops[i]["len"], loops.index))],
        )
        # quart1, quart2, quart3 = df["len"].quantile([0.25, 0.5, 0.75]).tolist()
        # return (
        #     df.loc[df["len"] <= quart1],
        #     df.loc[(quart1 < df["len"]) & (df["len"] <= quart2)],
        #     df.loc[(quart2 < df["len"]) & (df["len"] <= quart3)],
        #     df.loc[quart3 < df["len"]],
        # )

    def in_complete_loop(
        self, loop_df: pd.DataFrame[COL_START:float, COL_END:float] | None = None
    ) -> float:
        """Find single average c0 of loop cover in a chromosome."""
        # TODO: Use Loops instead of loop_df.
        if loop_df is None:
            loop_df = self._loops._loop_df

        return round(
            self._loops._chr.c0_spread()[self._loops.covermask(loop_df)].mean(), 3
        )

    def in_complete_non_loop(self) -> float:
        return round(NonCoverLoops(self._loops).mean_c0, 3)

    def in_quartile_by_len(self) -> list[float]:
        """Find average c0 of collection of loops by dividing them into
        quartiles by length"""
        quart_loops = self._get_quartile_dfs(self._loops)
        return list(map(lambda loops: CoverLoops(loops).mean_c0, quart_loops))

    def in_quartile_by_pos(self, loop_df: pd.DataFrame = None) -> np.ndarray:
        """Find average c0 of different positions in collection of loops

        Does not use loop cover.

        Returns:
            A 1D numpy array of size 4
        """
        if loop_df is None:
            loop_df = self._loops._loop_df

        chrv_c0_spread = self._loops._chr.c0_spread()

        def _avg_c0_in_quartile_by_pos(row: pd.Series) -> list[float]:
            quart_pos = row.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).astype(int)
            quart_range = list(
                map(
                    lambda idx: (quart_pos.iloc[idx], quart_pos.iloc[idx + 1]),
                    range(len(quart_pos) - 1),
                )
            )
            return list(
                map(lambda r: chrv_c0_spread[r[0] - 1 : r[1] - 1].mean(), quart_range)
            )

        result = np.round(
            np.mean(
                loop_df[["start", "end"]]
                .apply(_avg_c0_in_quartile_by_pos, axis=1)
                .tolist(),
                axis=0,
            ),
            3,
        )
        assert result.shape == (4,)
        return result

    def in_quartile_by_pos_in_quart_len(self) -> np.ndarray:
        """Find average c0 of different positions in quartile of loops by length

        Returns:
            A 1D numpy array of size 16.
        """
        loop_df = self._loops._loop_df
        quart_loop_df = self._get_quartile_dfs(loop_df)
        return np.array(list(map(self.in_quartile_by_pos, quart_loop_df))).flatten()

    def around_anc(
        self, pos: Literal["start", "end", "center"], lim: int = 500, loop_df=None
    ) -> float:
        if loop_df is None:
            loop_df = self._loops._loop_df

        chrv_c0_spread = self._loops._chr.c0_spread()

        loop_df = loop_df.assign(center=lambda df: (df["start"] + df["end"]) / 2)

        return round(
            sum(
                map(
                    lambda idx: chrv_c0_spread[
                        int(loop_df.iloc[idx][pos])
                        - lim
                        - 1 : int(loop_df.iloc[idx][pos])
                        + lim
                    ].mean(),
                    range(len(loop_df)),
                )
            )
            / len(loop_df),
            3,
        )

    def around_anc_in_quartile_by_len(
        self, pos: Literal["start", "end", "center"], lim: int = 500
    ) -> list[float]:
        """
        Find average c0 of collection of loops by dividing them into
        quartiles by length

        Returns:
            A list of 4 float numbers
        """
        quart_loop_df = self._get_quartile_dfs(self._loops)
        return list(map(self.around_anc, [pos] * 4, [lim] * 4, quart_loop_df))

    def in_nuc_linker(self, nuc_half: int = 73) -> tuple[float, float]:
        """
        Returns:
            A tuple: nuc mean C0, linker mean C0
        """
        nuc_cover = Nucleosomes(self._loops._chr).get_nuc_regions(nuc_half)

        loop_cover = self._loops.covermask(self._loops._loop_df)
        return (
            np.round(self._loops._chr.c0_spread()[loop_cover & nuc_cover].mean(), 3),
            np.round(self._loops._chr.c0_spread()[loop_cover & ~nuc_cover].mean(), 3),
        )

    def in_non_loop_nuc_linker(self, nuc_half: int = 73):
        nuc_cover = Nucleosomes(self._loops._chr).get_nuc_regions(nuc_half)

        loop_cover = self._loops.covermask(self._loops._loop_df)
        return (
            np.round(self._loops._chr.c0_spread()[~loop_cover & nuc_cover].mean(), 3),
            np.round(self._loops._chr.c0_spread()[~loop_cover & ~nuc_cover].mean(), 3),
        )


class MCMeanLoops:
    pass


class MultiChrmMeanLoopsCollector:
    """
    Class to accumulate various mean operations in loops in multiple chromosomes.

    Result is stored in a dataframe for side-by-side comparison. The dataframe
    contains a row for each chromosome.
    """

    # TODO: Use MultiChrm, MultiLoops
    OP_CHRM_MEAN = 0
    OP_CHRM_NUC_LINKER_MEAN = 1
    OP_LOOP_COVER_FRAC = 2
    OP_LOOP_MEAN = 3
    OP_LOOP_NUC_LINKER_MEAN = 4
    OP_NON_LOOP_MEAN = 5
    OP_NON_LOOP_NUC_LINKER_MEAN = 6
    OP_QUARTILE_BY_LEN = 7
    OP_QUARTILE_BY_POS = 8
    OP_QUARTILE_BY_LEN_POS = 9
    OP_ANCHOR_CENTER_BP = 10
    OP_QUARTILE_LEN_ANCHOR_CENTER_BP = 11
    OP_NUM_LOOPS = 12
    OP_NUM_LOOPS_LT_NON_LOOP = 13
    OP_NUM_LOOPS_L_LT_NLL = 14
    OP_NUM_LOOPS_N_LT_NLN = 15

    def __init__(
        self,
        prediction: Prediction,
        chrids: tuple[ChrId] = ChrIdList,
        mxlen: int | None = None,
    ):
        # TODO: Rename collector_df
        self._mcloop_df = pd.DataFrame({"ChrID": chrids})
        self._prediction = prediction
        self._chrs = self._get_chromosomes()

        self._mcloops = self._chrs.apply(lambda chrm: Loops(chrm, mxlen))
        self._mcmloops = self._mcloops.apply(lambda loops: MeanLoops(loops))
        self._mcnucs = self._chrs.apply(lambda chrm: Nucleosomes(chrm))
        self._mxlen = mxlen

    def __str__(self):
        ext = "with_vl" if "VL" in self._mcloop_df["ChrID"].tolist() else "without_vl"

        return f"md_{self._prediction}_mx_{self._mxlen}_{ext}"

    def _get_chromosomes(self) -> pd.Series:
        """Create a Pandas Series of Chromosomes"""
        return self._mcloop_df["ChrID"].apply(
            lambda id: Chromosome(id, self._prediction)
            if id != "VL"
            else Chromosome(id, None)
        )

    def _create_multiple_col(self, func: Callable, *args) -> np.ndarray:
        """
        Call functions of Loops in each chromosome and split result into
        multiple columns

        Returns: A 2D numpy array, where each column represents a column for
            dataframe
        """
        return np.array(
            self._mcmloops.apply(lambda mloops: func(mloops, *args)).tolist()
        )

    def _add_chrm_mean(self) -> None:
        self._mcloop_df["chromosome"] = self._chrs.apply(
            lambda chr: chr.c0_spread().mean()
        )

    def _add_chrm_nuc_linker_mean(self) -> None:
        nuc_linker_arr = np.array(
            self._mcnucs.apply(lambda nucs: nucs.find_avg_nuc_linker_c0()).tolist()
        )
        nuc_linker_cols = ["chrm_nuc", "chrm_linker"]
        self._mcloop_df[nuc_linker_cols] = pd.DataFrame(
            nuc_linker_arr, columns=nuc_linker_cols
        )

    def _add_loop_cover_frac(self) -> None:
        self._mcloop_df["cover"] = self._mcmloops.apply(
            lambda mloops: mloops._loops.covermask().mean()
        )

    def _add_loop_mean(self) -> None:
        self._mcloop_df["loop"] = self._mcmloops.apply(
            lambda mloops: mloops.in_complete_loop()
        )

    def _add_loop_nuc_linker_mean(self) -> None:
        loop_nuc_linker_cols = ["loop_nuc", "loop_linker"]
        self._mcloop_df[loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_nuc_linker),
            columns=loop_nuc_linker_cols,
        )

    def _add_non_loop_mean(self) -> Literal["non_loop"]:
        if "non_loop" not in self._mcloop_df.columns:
            self._mcloop_df["non_loop"] = self._mcmloops.apply(
                lambda mloops: mloops.in_complete_non_loop()
            )

        return "non_loop"

    def _add_non_loop_nuc_linker_mean(
        self,
    ) -> list[Literal["non_loop_nuc"], Literal["non_loop_linker"]]:
        non_loop_nuc_linker_cols = ["non_loop_nuc", "non_loop_linker"]
        self._mcloop_df[non_loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_non_loop_nuc_linker),
            columns=non_loop_nuc_linker_cols,
        )
        return non_loop_nuc_linker_cols

    def _add_quartile_by_len(self) -> None:
        quart_len_cols = ["quart_len_1", "quart_len_2", "quart_len_3", "quart_len_4"]
        self._mcloop_df[quart_len_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_quartile_by_len),
            columns=quart_len_cols,
        )

    def _add_quartile_by_pos(self) -> None:
        quart_pos_cols = ["quart_pos_1", "quart_pos_2", "quart_pos_3", "quart_pos_4"]
        self._mcloop_df[quart_pos_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_quartile_by_pos),
            columns=quart_pos_cols,
        )

    def _add_quartile_by_len_pos(self) -> None:
        quart_len_pos_cols = [
            f"quart_len_{p[0]}_pos_{p[1]}"
            for p in itertools.product(range(1, 5), range(1, 5))
        ]
        self._mcloop_df[quart_len_pos_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_quartile_by_pos_in_quart_len),
            columns=quart_len_pos_cols,
        )

    def _add_anchor_center_bp(self) -> None:
        def _get_anc_bp_it():
            return itertools.product(["start", "center", "end"], [500, 200, 50])

        anc_bp_cols = [f"{p[0]}_{p[1]}" for p in _get_anc_bp_it()]

        anc_bp_cols_df = pd.concat(
            [
                pd.Series(
                    self._mcmloops.apply(lambda mloops: mloops.around_anc(p[0], p[1])),
                    name=f"{p[0]}_{p[1]}",
                )
                for p in _get_anc_bp_it()
            ],
            axis=1,
        )
        self._mcloop_df[anc_bp_cols] = anc_bp_cols_df

    def _add_quartile_len_anchor_center_bp(self) -> None:
        quart_anc_len_cols = [
            f"{p[0]}_200_quart_len_{p[1]}"
            for p in itertools.product(["start", "center", "end"], range(1, 5))
        ]
        self._mcloop_df[quart_anc_len_cols] = pd.concat(
            [
                pd.DataFrame(
                    self._create_multiple_col(
                        MeanLoops.around_anc_in_quartile_by_len, pos, 200
                    ),
                    columns=quart_anc_len_cols[pos_idx * 4 : (pos_idx + 1) * 4],
                )
                for pos_idx, pos in enumerate(["start", "center", "end"])
            ],
            axis=1,
        )

    def _subtract_mean_chrm_c0(self) -> None:
        exclude_cols = ["chromosome", "ChrID"]
        if "cover" in self._mcloop_df.columns:
            exclude_cols += ["cover"]
        self._mcloop_df[
            self._mcloop_df.drop(columns=exclude_cols).columns
        ] = self._mcloop_df.drop(columns=exclude_cols).apply(
            lambda col: col - self._mcloop_df["chromosome"]
        )

    def _add_num_loops(self) -> Literal["num_loops"]:
        # TODO: Use wrapper hof, check cols
        self._mcloop_df["num_loops"] = self._mcmloops.apply(
            lambda mloops: len(mloops._loops._loop_df)
        )

        return "num_loops"

    def _add_num_loops_lt_non_loop(self) -> Literal["num_loops_lt_nl"]:
        # TODO: Use func id and col_for
        num_loops_lt_nl_col = "num_loops_lt_nl"

        if num_loops_lt_nl_col in self._mcloop_df.columns:
            return num_loops_lt_nl_col

        # Add mean c0 in each loop in loops object
        mean_cols = self._mcmloops.apply(
            lambda mloops: mloops._loops.add_mean_c0()
        ).iloc[0]

        # Add non loop mean of each chromosome to this object
        non_loop_col = self._add_non_loop_mean()

        # Compare mean c0 of whole loop and non loop
        mcmloops = pd.Series(self._mcmloops, name="mcmloops")
        self._mcloop_df[num_loops_lt_nl_col] = pd.DataFrame(mcmloops).apply(
            lambda mloops: (
                mloops["mcmloops"]._loops._loop_df[mean_cols[0]]
                < self._mcloop_df.iloc[mloops.name][non_loop_col]
            ).sum(),
            axis=1,
        )

        return num_loops_lt_nl_col

    def _add_num_loops_l_lt_nll(self) -> Literal["num_loops_l_lt_nll"]:
        func_id = 14
        if self.col_for(func_id) in self._mcloop_df.columns:
            return self.col_for(func_id)

        # Add mean c0 in each loop in loops object
        mean_cols = self._mcmloops.apply(
            lambda mloops: mloops._loops.add_mean_c0()
        ).iloc[0]

        # Add non loop nuc and linker mean of each chromosome to this object
        _, nl_l_col = self._add_non_loop_nuc_linker_mean()

        # Compare mean c0 of linkers in loop and non loop linkers
        mcmloops = pd.Series(self._mcmloops, name="mcmloops")
        self._mcloop_df[self.col_for(func_id)] = pd.DataFrame(mcmloops).apply(
            lambda mloops: (
                mloops["mcmloops"]._loops._loop_df[mean_cols[2]]
                < self._mcloop_df.iloc[mloops.name][nl_l_col]
            ).sum(),
            axis=1,
        )

        return self.col_for(func_id)

    def _add_num_loops_n_lt_nln(self) -> Literal["num_loops_n_lt_nln"]:
        # TODO: Refactor -> reduce duplicate code
        func_id = self.OP_NUM_LOOPS_N_LT_NLN
        if self.col_for(func_id) in self._mcloop_df.columns:
            return self.col_for(func_id)

        # Add mean c0 in each loop in loops object
        mean_cols = self._mcmloops.apply(
            lambda mloops: mloops._loops.add_mean_c0()
        ).iloc[0]

        # Add non loop nuc and linker mean of each chromosome to this object
        nl_n_col, _ = self._add_non_loop_nuc_linker_mean()

        # Compare mean c0 of nucs in loop and non loop nucs
        mcmloops = pd.Series(self._mcmloops, name="mcmloops")
        self._mcloop_df[self.col_for(func_id)] = pd.DataFrame(mcmloops).apply(
            lambda mloops: (
                mloops["mcmloops"]._loops._loop_df[mean_cols[1]]
                < self._mcloop_df.iloc[mloops.name][nl_n_col]
            ).sum(),
            axis=1,
        )

        return self.col_for(func_id)

    def col_for(self, func_id: int) -> str | list[str]:
        col_map = {
            self.OP_NON_LOOP_NUC_LINKER_MEAN: ["non_loop_nuc", "non_loop_linker"],
            self.OP_NUM_LOOPS: "num_loops",
            self.OP_NUM_LOOPS_LT_NON_LOOP: "num_loops_lt_nl",
            self.OP_NUM_LOOPS_L_LT_NLL: "num_loops_l_lt_nll",
            self.OP_NUM_LOOPS_N_LT_NLN: "num_loops_n_lt_nln",
        }
        return col_map[func_id]

    def save_avg_c0_stat(
        self, mean_methods: list[int] | None = None, subtract_chrm=True
    ) -> Path:
        """
        Args:
            mean_methods: A list of int. Between 0-11 (inclusive)
            subtract_chrm: Whether to subtract chrm c0

        Returns:
            The path where dataframe is saved
        """
        method_map = {
            self.OP_CHRM_MEAN: self._add_chrm_mean,
            self.OP_CHRM_NUC_LINKER_MEAN: self._add_chrm_nuc_linker_mean,
            self.OP_LOOP_COVER_FRAC: self._add_loop_cover_frac,
            self.OP_LOOP_MEAN: self._add_loop_mean,
            self.OP_LOOP_NUC_LINKER_MEAN: self._add_loop_nuc_linker_mean,
            self.OP_NON_LOOP_MEAN: self._add_non_loop_mean,
            self.OP_NON_LOOP_NUC_LINKER_MEAN: self._add_non_loop_nuc_linker_mean,
            self.OP_QUARTILE_BY_LEN: self._add_quartile_by_len,
            self.OP_QUARTILE_BY_POS: self._add_quartile_by_pos,
            self.OP_ANCHOR_CENTER_BP: self._add_anchor_center_bp,
            self.OP_QUARTILE_LEN_ANCHOR_CENTER_BP: self._add_quartile_len_anchor_center_bp,
            self.OP_NUM_LOOPS: self._add_num_loops,
            self.OP_NUM_LOOPS_LT_NON_LOOP: self._add_num_loops_lt_non_loop,
            self.OP_NUM_LOOPS_L_LT_NLL: self._add_num_loops_l_lt_nll,
            self.OP_NUM_LOOPS_N_LT_NLN: self._add_num_loops_n_lt_nln,
        }

        if mean_methods is None:
            mean_methods = list(method_map.keys())

        for m in mean_methods:
            method_map[m]()

        if subtract_chrm:
            self._subtract_mean_chrm_c0()

        self._mcloop_df["model"] = np.full(
            (len(self._mcloop_df),), str(self._prediction)
        )

        save_df_path = f"{PathObtain.data_dir()}/generated_data/mcloops/multichr_avg_c0_stat_{self}.tsv"
        FileSave.append_tsv(self._mcloop_df, save_df_path)
        return save_df_path

    def plot_scatter_loop_nuc_linker_mean(self):
        self.save_avg_c0_stat([3, 4, 5, 6], subtract_chrm=False)
        labels = [
            "loop",
            "loop_nuc",
            "loop_linker",
            "non_loop",
            "non_loop_nuc",
            "non_loop_linker",
        ]

        # Show grid below other plots
        plt.rc("axes", axisbelow=True)

        arr = self._mcloop_df[labels].values
        x = np.arange(arr.shape[0])
        markers = ["o", "s", "p", "P", "*", "D"]
        for i in range(arr.shape[1]):
            plt.scatter(x, arr[:, i], marker=markers[i], label=labels[i])

        plt.grid()
        plt.xticks(x, self._mcloop_df["ChrID"])
        plt.xlabel("Chromosome")
        plt.ylabel("Mean C0")
        plt.title(
            "Comparison of mean C0 in nucleosome and linker region in loop"
            f" vs. non-loop with max loop length = {self._mxlen}"
        )
        plt.legend()

        FileSave.figure(f"{PathObtain.figure_dir()}/mcloops/nuc_linker_mean_{self}.png")

    def plot_loop_cover_frac(self) -> Path:
        self.save_avg_c0_stat([2], subtract_chrm=False)
        cv_frac = self._mcloop_df["cover"].to_numpy() * 100
        x = np.arange(cv_frac.size)
        plt.bar(x, cv_frac)
        plt.grid()
        plt.xticks(x, self._mcloop_df["ChrID"])
        plt.xlabel("Chromosome")
        plt.ylabel("Loop cover (%)")
        plt.title(
            f"Loop cover percentage in whole chromosome with max length = {self._mxlen}"
        )
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcloop/loop_cover_{self}.png"
        )

    def get_loops_data(self) -> pd.DataFrame:
        """Get data of all loops"""
        self._mcloops.apply(lambda loops: loops.add_mean_c0())
        all_loops_data = self._mcloops.apply(
            lambda loops: [loops[i] for i in range(len(loops))]
        )

        return pd.DataFrame(functools.reduce(operator.add, all_loops_data.tolist()))


class MultiChrmMeanLoopsAggregator:
    """
    Class to find aggregated statistics of loops in all chromosomes
    """

    def __init__(self, coll: MultiChrmMeanLoopsCollector):
        self._coll = coll
        self._agg_df = pd.DataFrame({"ChrIDs": [coll._mcloop_df["ChrID"].tolist()]})

    def _loop_lt_nl(self):
        self._coll.save_avg_c0_stat(
            [self._coll.OP_NUM_LOOPS, self._coll.OP_NUM_LOOPS_LT_NON_LOOP], False
        )
        lp_lt_nl = (
            self._coll._mcloop_df[
                self._coll.col_for(self._coll.OP_NUM_LOOPS_LT_NON_LOOP)
            ].sum()
            / self._coll._mcloop_df[self._coll.col_for(self._coll.OP_NUM_LOOPS)].sum()
        )
        self._agg_df["loop_lt_nl"] = lp_lt_nl * 100

    def _loop_l_lt_nll(self):
        self._coll.save_avg_c0_stat(
            [self._coll.OP_NUM_LOOPS, self._coll.OP_NUM_LOOPS_L_LT_NLL], False
        )
        self._agg_df["loop_l_lt_nll"] = (
            self._coll._mcloop_df[
                self._coll.col_for(self._coll.OP_NUM_LOOPS_L_LT_NLL)
            ].sum()
            / self._coll._mcloop_df[self._coll.col_for(self._coll.OP_NUM_LOOPS)].sum()
            * 100
        )

    def _loop_l_lt_nln(self):
        self._coll.save_avg_c0_stat(
            [self._coll.OP_NUM_LOOPS, self._coll.OP_NUM_LOOPS_N_LT_NLN], False
        )
        self._agg_df["loop_n_lt_nln"] = (
            self._coll._mcloop_df[
                self._coll.col_for(self._coll.OP_NUM_LOOPS_N_LT_NLN)
            ].sum()
            / self._coll._mcloop_df[self._coll.col_for(self._coll.OP_NUM_LOOPS)].sum()
            * 100
        )

    def save_stat(self, methods: list[int]) -> Path:
        method_map = {
            0: self._loop_lt_nl,
            1: self._loop_l_lt_nll,
            2: self._loop_l_lt_nln,
        }

        for m in methods:
            method_map[m]()

        save_df_path = (
            f"{PathObtain.data_dir()}/generated_data/mcloops/agg_stat_{self._coll}.tsv"
        )
        return FileSave.append_tsv(self._agg_df, save_df_path)

    def scatter_plot_c0_vs_loop_size(self) -> Path:
        all_loops_df = self._coll.get_loops_data()
        x = all_loops_df[COL_LEN]
        y = all_loops_df[COL_MEAN_C0_FULL]
        PlotUtil.show_grid()
        plt.scatter(x, y)
        plt.xscale("log")

        plt.xlabel("Loop length in bp (logarithmic)")
        plt.ylabel("Mean C0")
        plt.title("Mean C0 vs Loop Size")
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/mcloops/c0_vs_loop_size_{self._coll}.png"
        )
