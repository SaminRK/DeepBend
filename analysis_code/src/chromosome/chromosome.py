from __future__ import annotations
import math
from pathlib import Path
import time
from typing import Iterable, Literal, Union, Any
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from nptyping import NDArray

from models.prediction import Prediction
from util.reader import DNASequenceReader, SEQ_NUM_COL, SEQ_COL
from util.constants import CHRVL, SEQ_LEN, ChrIdList
from util.custom_types import ChrId, PosOneIdx, YeastChrNum, PositiveInt, C0
from util.util import Attr, DataCache, FileSave, PathObtain, PlotUtil


class C0Spread:
    mean7 = "mean7"
    mcvr = "mcvr"
    wt = "wt"
    sng = "sng"


# TODO: Tell story top-down following newspaper metaphor

# TODO: Rename Spread to ChrmSpread
class Spread:
    """Spread C0 at each bp from C0 of 50-bp sequences at 7-bp resolution"""

    def __init__(self, seq_c0_res7: np.ndarray, id: ChrId, model_no: int = 6):
        """
        Construct a spread object

        Args:
            seq_c0_res7: C0 of 50-bp sequences at 7-bp resolution
            id: Chromosome ID
        """
        self._seq_c0_res7 = seq_c0_res7
        self.total_bp = ChrmCalc.total_bp(seq_c0_res7.size)
        self.id = id
        self._model_no = model_no

    def _covering_sequences_at(self, pos: int) -> np.ndarray:
        """
        Find covering 50-bp sequences in chr library of a bp in chromosome.

        Args:
            pos: position in chr (1-indexed)

        Returns:
            A numpy 1D array containing sequence numbers of chr library
            in increasing order
        """
        if pos < SEQ_LEN:  # 1, 2, ... , 50
            arr = np.arange((pos + 6) // 7) + 1
        elif pos > self.total_bp - SEQ_LEN:  # For chr V, 576822, ..., 576871
            arr = (
                -(np.arange((self.total_bp - pos + 1 + 6) // 7)[::-1])
                + self._seq_c0_res7.size
            )
        elif pos % 7 == 1:  # For chr V, 50, 57, 64, ..., 576821
            arr = np.arange(8) + (pos - SEQ_LEN + 7) // 7
        else:
            arr = np.arange(7) + (pos - SEQ_LEN + 14) // 7

        start_pos = (arr - 1) * 7 + 1
        end_pos = start_pos + SEQ_LEN - 1
        assert np.all(pos >= start_pos) and np.all(pos <= end_pos)

        return arr

    def _mean_of_7(self) -> np.ndarray:
        """
        Determine C0 at each bp by taking mean of 7 covering sequences

        If 8 sequences cover a bp, first 7 considered. If less than 7 seq
        cover a bp, nearest 7-seq mean is used.
        """
        saved_data = Path(
            f"{PathObtain.data_dir()}/generated_data/spread/spread_c0_mean7_{self.id}_m_{self._model_no}.tsv"
        )
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep="\t")["c0_mean7"].to_numpy()

        mvavg = ChrmCalc.moving_avg(self._seq_c0_res7, 7)
        spread_mvavg = np.vstack(
            (mvavg, mvavg, mvavg, mvavg, mvavg, mvavg, mvavg)
        ).ravel(order="F")
        full_spread = np.concatenate(
            (
                np.full((42,), spread_mvavg[0]),
                spread_mvavg,
                np.full((43,), spread_mvavg[-1]),
            )
        )
        assert full_spread.shape == (self.total_bp,)

        FileSave.tsv(
            pd.DataFrame(
                {"position": np.arange(self.total_bp) + 1, "c0_mean7": full_spread}
            ),
            saved_data,
        )
        return full_spread

    def _mean_of_covering_seq(self) -> np.ndarray:
        """Determine C0 at each bp by average of covering 50-bp sequences around"""

        def _balanced_c0_at(pos) -> float:
            seq_indices = self._covering_sequences_at(pos) - 1
            return self._seq_c0_res7[seq_indices].mean()

        def _c0_of_cover() -> pd.DataFrame:
            t = time.time()
            res = np.array(list(map(_balanced_c0_at, np.arange(self.total_bp) + 1)))
            print("Calculation of spread c0 balanced:", time.time() - t, "seconds.")
            return pd.DataFrame(
                {"position": np.arange(self.total_bp) + 1, "c0_balanced": res}
            )

        return DataCache.calc_df_tsv(
            f"spread/spread_c0_balanced_{self.id}_m_{self._model_no}.tsv", _c0_of_cover
        )["c0_balanced"].to_numpy()

    def _weighted_covering_seq(self) -> np.ndarray:
        """Determine C0 at each bp by weighted average of covering 50-bp sequences around"""
        saved_data = Path(
            f"{PathObtain.data_dir()}/generated_data/spread/spread_c0_weighted_{self.id}_m_{self._model_no}.tsv"
        )
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep="\t")["c0_weighted"].to_numpy()

        def weights_for(size: int) -> list[int]:
            # TODO: Use short algorithm
            if size == 1:
                return [1]
            elif size == 2:
                return [1, 1]
            elif size == 3:
                return [1, 2, 1]
            elif size == 4:
                return [1, 2, 2, 1]
            elif size == 5:
                return [1, 2, 3, 2, 1]
            elif size == 6:
                return [1, 2, 3, 3, 2, 1]
            elif size == 7:
                return [1, 2, 3, 4, 3, 2, 1]
            elif size == 8:
                return [1, 2, 3, 4, 4, 3, 2, 1]

        def weighted_c0_at(pos) -> float:
            seq_indices = self._covering_sequences_at(pos) - 1
            c0s = self._seq_c0_res7[seq_indices]
            return np.sum(c0s * weights_for(c0s.size)) / sum(weights_for(c0s.size))

        t = time.time()
        res = np.array(list(map(weighted_c0_at, np.arange(self.total_bp) + 1)))
        print(print("Calculation of spread c0 weighted:", time.time() - t, "seconds."))

        # Save data
        if not saved_data.parents[0].is_dir():
            saved_data.parents[0].mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"position": np.arange(self.total_bp) + 1, "c0_weighted": res}
        ).to_csv(saved_data, sep="\t", index=False)

        return res

    def _from_single_seq(self) -> np.ndarray:
        """Determine C0 at each bp by spreading C0 of a 50-bp seq to position 22-28"""
        c0_arr = self._seq_c0_res7
        spread = np.concatenate(
            (
                np.full((21,), c0_arr[0]),
                np.vstack(([c0_arr] * 7)).ravel(order="F"),
                np.full((22,), c0_arr[-1]),
            )
        )
        # assert spread.size == self.total_bp

        return spread

    def c0_spread(self, sprd: C0Spread) -> np.ndarray:
        d = {
            C0Spread.mean7: self._mean_of_7,
            C0Spread.mcvr: self._mean_of_covering_seq,
            C0Spread.wt: self._weighted_covering_seq,
            C0Spread.sng: self._from_single_seq,
        }
        return d[sprd]()


class ChrmCalc:
    @classmethod
    def moving_avg(self, arr: NDArray[(Any,)], k: int) -> NDArray[(Any,)]:
        assert len(arr.shape) == 1

        # Find first MA
        ma = np.array([arr[:k].mean()])

        # Iteratively find next MAs
        for i in range(arr.size - k):
            ma = np.append(ma, ma[-1] + (arr[i + k] - arr[i]) / k)

        assert len(ma) == len(arr) - k + 1
        return ma

    @classmethod
    def total_bp(self, num_seq: int):
        return (num_seq - 1) * 7 + SEQ_LEN


C0 = "c0"


class Chromosome:
    "Class that holds sequence and C0 array of a yeast chromosome"

    def __init__(
        self,
        id: Union[YeastChrNum, Literal["VL"]],
        prediction: Prediction | None = None,
        spread_str: C0Spread = "mean7",
    ):
        """
        Create a Chromosome object

        Args:
            id: For Roman Numbers, predicted C0 is used. 'VL' represents
                chr V library of bendability data.
            spread_str: Which type of spread to use.
        """
        self.id = id

        if id == "VL":
            self._prediction = None
            self._df = DNASequenceReader().get_processed_data()[CHRVL]
        else:
            self._prediction = prediction or Prediction(30)
            self._df = self._get_chrm_df()

        self.spread_str = spread_str
        self._mean_c0 = None
        self._seq = None

    def __str__(self):
        return f"chrm_s_{self.spread_str}_m_{self.predict_model_no()}_{self.id}"

    @property
    def mean_c0(self):
        return Attr.calc_attr(self, "_mean_c0", lambda: self.c0_spread().mean())

    @property
    def total_bp(self):
        return ChrmCalc.total_bp(len(self._df))

    @property
    def c0_type(self):
        return "actual" if self.id == "VL" else "predicted"

    @property
    def number(self) -> YeastChrNum:
        return "V" if self.id == "VL" else self.id

    @property
    def seq(self) -> str:
        def _seq() -> str:
            return self._df[SEQ_COL][0] + "".join(self._df[SEQ_COL][1:].str[-7:])

        return Attr.calc_attr(self, "_seq", _seq)

    def seqf(
        self,
        start: PosOneIdx | Iterable[PosOneIdx],
        end: PosOneIdx | Iterable[PosOneIdx],
    ) -> str | list[str]:
        assert type(start) == type(end)
        if type(start) == PosOneIdx or type(start) == float:
            return self.seq[int(start) - 1 : int(end)]
        elif isinstance(start, Iterable):
            return list(
                map(
                    lambda s, e: self.seq[s - 1 : e],
                    np.array(start).astype(int),
                    np.array(end).astype(int),
                )
            )

    def _get_chrm_df(self) -> pd.DataFrame[SEQ_NUM_COL:int, SEQ_COL:str, C0:float]:
        def _chrm_df():
            df = DNASequenceReader().read_yeast_genome(self.number)
            return self._prediction.predict(df).rename(columns={"c0_predict": "C0"})

        return DataCache.calc_df_tsv(
            f"predictions/chr{self.number}_pred_m_{self._prediction._model_no}.tsv",
            _chrm_df,
        )

    def read_chr_lib_segment(self, start: int, end: int) -> pd.DataFrame:
        """
        Get sequences in library of a chromosome segment.

        Returns:
            A pandas.DataFrame containing chr library of selected segment.
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil((end - SEQ_LEN + 1) / 7)

        return self._df.loc[
            (self._df["Sequence #"] >= first_seq_num)
            & (self._df["Sequence #"] <= last_seq_num),
            :,
        ]

    def horizline(self, *args) -> None:
        # TODO: Remove middle
        PlotUtil.avg_horizline(*args)

    def plot_moving_avg(self, start: int, end: int, plotnuc: bool = False) -> None:
        """
        Plot C0, a few moving averages of C0 and nuc. centers in a segment of chr.

        Does not give labels so that custom labels can be given after calling this
        function.

        Args:
            start: Start position in the chromosome
            end: End position in the chromosome
        """
        PlotUtil.clearfig()
        PlotUtil.show_grid()

        x = np.arange(start - 1, end)
        y = self.c0_spread()[start - 1 : end]
        plt.plot(x, y, color="blue", alpha=0.5, label=1)

        k = [10]  # , 25, 50]
        colors = ["green"]  # , 'red', 'black']
        alpha = [0.7]  # , 0.8, 0.9]
        for p in zip(k, colors, alpha):
            ma = ChrmCalc.moving_avg(y, p[0])
            plt.plot(
                (x + ((p[0] - 1) * 7) // 2)[: ma.size],
                ma,
                color=p[1],
                alpha=p[2],
                label=p[0],
            )

        PlotChrm(self).plot_avg()

        if plotnuc:
            nuc_df = DNASequenceReader().read_nuc_center()
            centers = nuc_df.loc[nuc_df["Chromosome ID"] == f"chr{self.number}"][
                "Position"
            ].to_numpy()
            centers = centers[start < centers < end]

            for c in centers:
                plt.axvline(x=c, color="grey", linestyle="--")

        plt.legend()

    def plot_c0(self, start: int, end: int) -> None:
        """Plot C0, moving avg., nuc. centers of a segment in chromosome
        and add appropriate labels.
        """
        self.plot_moving_avg(start, end)

        plt.ylim(-0.8, 0.6)
        plt.xlabel(f"Position along chromosome {self.number}")
        plt.ylabel("Moving avg. of C0")
        plt.title(
            f"C0, 10-bp moving avg. of C0 and nuclesome centers in Chr {self.number} ({start}-{end})"
        )

        # Save figure
        plt.gcf().set_size_inches(12, 6)
        ma_fig_dir = f"{PathObtain.figure_dir()}/chromosome/{self.id}"
        if not Path(ma_fig_dir).is_dir():
            Path(ma_fig_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{ma_fig_dir}/ma_{start}_{end}_s_{self.spread_str}_m_{self._prediction._model_no}.png",
            dpi=200,
        )
        plt.show()

    # TODO: Make spread, predict model no -> property
    def c0_spread(self) -> np.ndarray:
        def _c0():
            return Spread(
                self._df["C0"].values, self.id, self.predict_model_no()
            ).c0_spread(self.spread_str)

        return Attr.calc_attr(self, "_c0_spread", _c0)

    def predict_model_no(self) -> int:
        return self._prediction._model_no if self.id != "VL" else None

    def mean_c0_around_bps(
        self,
        bps: np.ndarray | list[int] | pd.Series,
        neg_lim: PositiveInt,
        pos_lim: PositiveInt,
    ) -> np.ndarray:
        """
        Aggregate mean c0 in each bp of some equal-length segments.

        Segments are defined by limits from positions in input `bps` array.

        Args:
            bps: An 1D Numpy array of bp to define segment. (1-indexed)
        """
        assert neg_lim >= 0 and pos_lim >= 0
        result = np.array(
            [self.c0_spread()[bp - 1 - neg_lim : bp + pos_lim] for bp in bps]
        ).mean(axis=0)
        assert result.shape == (neg_lim + pos_lim + 1,)
        return result

    def get_cvr_mask(
        self,
        bps: Iterable[PosOneIdx],
        neg_lim: PositiveInt,
        pos_lim: PositiveInt,
    ) -> np.ndarray:
        """
        Create a mask to cover segments that are defined by a bp and two
        limits.
        """
        cvr_arr = np.full((self.total_bp,), False)
        for bp in bps:
            cvr_arr[bp - 1 - neg_lim : bp + pos_lim] = True

        return cvr_arr

    def mean_c0_segment(self, start: PosOneIdx, end: PosOneIdx) -> float:
        return self.c0_spread()[start - 1 : end].mean()

    def mean_c0_of_segments(
        self,
        bps: np.ndarray | list[int] | pd.Series,
        neg_lim: PositiveInt,
        pos_lim: PositiveInt,
    ) -> float:
        """
        Find single mean c0 of covered region by segments.

        Segments are defined by a bp and two limits
        """
        # TODO: Support one limit
        cvr = self.get_cvr_mask(bps, neg_lim, pos_lim)
        result = self.c0_spread()[cvr].mean()
        return result

    def mean_c0_at_bps(
        self,
        bps: Iterable[PosOneIdx],
        neg_lim: PositiveInt,
        pos_lim: PositiveInt,
    ) -> NDArray[(Any,)]:
        """
        Find mean c0 of each segment defined by bps

        bp = 1-idxed
        segmnt len = neglim + poslim + 1
        """
        result = np.array(
            list(
                map(
                    lambda bp: self.c0_spread()[bp - 1 - neg_lim : bp + pos_lim].mean(),
                    np.array(bps),
                )
            )
        )

        assert result.shape == (len(bps),)
        return result


class PlotChrm:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0(self, start: PosOneIdx, end: PosOneIdx) -> None:
        x = np.arange(start, end + 1)
        y = self._chrm.c0_spread()[start - 1 : end]
        plt.plot(x, y)
        self.plot_avg()

    def plot_avg(self) -> None:
        """
        Plot a horizontal red line denoting avg. C0 in whole chromosome

        Best to draw after x limit is set.
        """
        PlotUtil.avg_horizline(self._chrm.mean_c0)


class MultiChrm:
    def __init__(self, chrmids: tuple[ChrId] = ChrIdList):
        self._chrmids = chrmids
        self._chrms = list(map(lambda id: Chromosome(id), chrmids))

    def __iter__(self) -> Iterable[Chromosome]:
        return iter(self._chrms)

    def __str__(self):
        if set(self._chrmids) == set(ChrIdList):
            return "all_pred"

        if set(self._chrmids) == set(("VL",)):
            return "VL"

        return ""


class ChrmOperator:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def c0_rgns(
        self,
        starts: np.ndarray | list[PosOneIdx] | pd.Series,
        ends: np.ndarray | list[PosOneIdx] | pd.Series,
    ) -> NDArray[(Any, Any), C0]:
        """
        Find c0 at each bp of regions. regions are equal len.
        """
        result = np.array(
            [self.c0(s, e) for s, e in zip(np.array(starts), np.array(ends))]
        )
        assert result.shape == (
            len(starts),
            np.array(ends)[0] - np.array(starts)[0] + 1,
        )
        return result

    def c0(
        self, start: float | PosOneIdx, end: float | PosOneIdx
    ) -> NDArray[(Any,), float]:
        return self._chrm.c0_spread()[int(start - 1) : int(end)]

    def mean_c0_regions_indiv(
        self, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> NDArray[(Any,), float]:
        assert all(starts > 0)
        assert all(ends >= starts)
        return np.array(
            list(
                map(
                    lambda se: self._chrm.c0_spread()[se[0] - 1 : se[1]].mean(),
                    zip(starts, ends),
                )
            )
        )

    def create_cover_mask(
        self, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> NDArray[(Any,), bool]:
        cvrmask = np.full((self._chrm.total_bp,), False)
        for s, e in zip(starts, ends):
            cvrmask[s - 1 : e] = True

        return cvrmask

    def in_lim(self, arr: Iterable[PosOneIdx], l: int) -> NDArray[(Any,), PosOneIdx]:
        bps = pd.Series(arr)
        return bps.loc[(l < bps) & (bps <= (self._chrm.total_bp - l))].to_numpy()
