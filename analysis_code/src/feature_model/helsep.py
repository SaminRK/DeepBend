from __future__ import annotations
import itertools as it
from pathlib import Path
from enum import Enum
from typing import Any, Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray

from util.kmer import KMer
from util.util import FileSave, PathObtain, get_possible_seq, gen_random_sequences, Attr
from util.reader import SEQ_COL, C0_COL
from util.custom_types import DNASeq, DiNc

# Constants
NUM_DINC_PAIRS = 136  # Possible dinucleotide pairs
NUM_DISTANCES = 48  # Possible distances between two dinucleotides
SEQ_COL = "Sequence"


class DincUtil:
    @classmethod
    def pairs_all(cls) -> list[tuple[DiNc, DiNc]]:
        """
        Generates all possible dinucleotide pairs
        """

        def _pairs():
            all_dinc = get_possible_seq(2)

            dinc_pairs = [pair for pair in it.combinations(all_dinc, 2)] + [
                (dinc, dinc) for dinc in all_dinc
            ]
            assert len(dinc_pairs) == 136

            return dinc_pairs

        return Attr.calc_attr(cls, "_dinc_pairs", _pairs)

    @classmethod
    def pair_str(cls, dinca: DiNc, dincb: DiNc) -> str:
        if dinca <= dincb:
            return dinca + "-" + dincb

        return cls.pair_str(dincb, dinca)


class HSAggr(Enum):
    MAX = "max"
    SUM = "sum"


class HelSep:
    """
    Helper class for counting helical separation

    The hel sep of a given NN-NN pair in a given sequence = Sum_{i = 10, 20, 30}
    max(p(i-1), p(i), p(i+1)) - Sum_{i = 5, 15, 25} max(p(i-1), p(i), p(i+1))

    where p(i) is the pairwise distance distribution function, the number of
    times that the two dinucleotides in the pair are separated by a distance i
    in the sequence, normalized by an expected p(i) for the NN-NN in random
    seqs.
    """

    def __init__(self):
        self._expected_dist_file = (
            f"{PathObtain.data_dir()}/generated_data/helical_separation/expected_p.tsv"
        )
        self._dinc_pairs = DincUtil.pairs_all()

    def helical_sep_of(
        self, seq_list: list[DNASeq], aggr: HSAggr = HSAggr.MAX
    ) -> pd.DataFrame:
        """
        Count normalized helical separation
        """
        pair_dist_occur = self._dinc_pair_dist_occur_normd(seq_list)

        def _dist_occur_max_at(pos: int) -> NDArray[(Any, 136), float]:
            return np.max(pair_dist_occur[:, :, pos - 2 : pos + 1], axis=2)

        def _dist_occur_sum_at(pos: int) -> NDArray[(Any, 136), float]:
            return np.sum(pair_dist_occur[:, :, pos - 2 : pos + 1], axis=2)

        if aggr == HSAggr.MAX:
            dist_occur_aggr = _dist_occur_max_at
        elif aggr == HSAggr.SUM:
            dist_occur_aggr = _dist_occur_sum_at

        at_hel_dist = sum(list(map(dist_occur_aggr, [10, 20, 30])))
        at_half_hel_dist = sum(list(map(dist_occur_aggr, [5, 15, 25])))

        dinc_df = pd.DataFrame(
            at_hel_dist - at_half_hel_dist,
            columns=list(
                map(
                    DincUtil.pair_str,
                    [p[0] for p in self._dinc_pairs],
                    [p[1] for p in self._dinc_pairs],
                )
            ),
        )
        dinc_df[SEQ_COL] = seq_list
        return dinc_df

    def _dinc_pair_dist_occur_normd(
        self, seq_list: list[DNASeq]
    ) -> NDArray[(Any, 136, 48), float]:
        """
        Calculates normalized p(i) for i = 1-48 for all dinc pairs
        """
        pair_dist_occur = np.array(list(map(self._pair_dinc_dist_in, seq_list)))
        assert pair_dist_occur.shape == (len(seq_list), 136, 48)

        exp_dist_occur = self.calculate_expected_p().drop(columns="Pair").values
        assert exp_dist_occur.shape == (136, 48)

        return pair_dist_occur / exp_dist_occur

    def calculate_expected_p(self) -> pd.DataFrame:
        """
        Calculates expected p(i) of dinucleotide pairs.
        """
        if Path(self._expected_dist_file).is_file():
            return pd.read_csv(self._expected_dist_file, sep="\t")

        # Generate 10000 random sequences
        seq_list = gen_random_sequences(10000)

        # Count mean distance for 136 pairs in generated sequences
        pair_all_dist = np.array(list(map(self._pair_dinc_dist_in, seq_list)))
        mean_pair_dist = pair_all_dist.mean(axis=0)
        assert mean_pair_dist.shape == (136, 48)

        # Save a dataframe of 136 rows x 49 columns
        df = pd.DataFrame(mean_pair_dist, columns=np.arange(48) + 1)
        df["Pair"] = list(map(lambda p: f"{p[0]}-{p[1]}", self._dinc_pairs))
        FileSave.tsv(df, self._expected_dist_file)
        return df

    def _pair_dinc_dist_in(
        self, seq: DNASeq
    ) -> NDArray[(NUM_DINC_PAIRS, NUM_DISTANCES), int]:
        """
        Find unnormalized p(i) for i = 1-48 for all dinucleotide pairs
        """
        pos_dinc = dict()
        for dinc in get_possible_seq(2):
            pos_dinc[dinc] = KMer.find_pos(dinc, seq)

        dinc_dists: list[list[int]] = map(
            lambda p: self._find_pair_dist(pos_dinc[p[0]], pos_dinc[p[1]]),
            self._dinc_pairs,
        )

        return np.array(
            list(
                map(
                    lambda one_pair_dist: np.bincount(
                        one_pair_dist, minlength=NUM_DISTANCES + 1
                    )[1:],
                    dinc_dists,
                )
            )
        )

    @classmethod
    def _find_pair_dist(
        cls, pos_one: Iterable[int], pos_two: Iterable[int]
    ) -> list[int]:
        """
        Find absolute distances from positions

        Example - Passing parameters [3, 5] and [1, 2] will return [2, 1, 4, 3]
        """
        return [
            abs(pos_pair[0] - pos_pair[1]) for pos_pair in it.product(pos_one, pos_two)
        ]


class HelSepPlot:
    @classmethod
    def plot_normalized_dist(
        cls, df: pd.DataFrame[SEQ_COL:str, C0_COL:float], library_name: str
    ) -> None:
        """
        Plots avg. normalized distance of sequences with most and least 1000 C0 values
        """
        hs = HelSep()

        least_1000 = df.sort_values(C0_COL).iloc[:1000]
        most_1000 = df.sort_values(C0_COL).iloc[-1000:]

        most1000_dist = hs._dinc_pair_dist_occur_normd(
            most_1000[SEQ_COL].tolist()
        ).mean(axis=0)
        least1000_dist = hs._dinc_pair_dist_occur_normd(
            least_1000[SEQ_COL].tolist()
        ).mean(axis=0)

        assert most1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)
        assert least1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)

        for i in range(NUM_DINC_PAIRS):
            plt.close()
            plt.clf()
            pair_str = DincUtil.pair_str(*DincUtil.pairs_all()[i])

            plt.plot(
                np.arange(NUM_DISTANCES) + 1,
                most1000_dist[i],
                linestyle="-",
                color="r",
                label="1000 most loopable sequences",
            )
            plt.plot(
                np.arange(NUM_DISTANCES) + 1,
                least1000_dist[i],
                linestyle="-",
                color="b",
                label="1000 least loopable sequences",
            )
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20))
            plt.xlabel("Position (bp)")
            plt.ylabel("Pairwise distance distribution, p(i)")
            plt.title(pair_str)
            FileSave.figure_in_figdir(
                f"distances/{library_name}/{pair_str}.png", bbox_inches="tight"
            )
