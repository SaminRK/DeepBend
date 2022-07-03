from __future__ import annotations
from typing import Iterable, Any

import regex as re
import numpy as np
from nptyping import NDArray

from util.util import rev_comp, get_possible_seq
from util.custom_types import DNASeq, KMerSeq


class KMer:
    @classmethod
    def all(cls, n: int) -> list[str]:
        return get_possible_seq(n)

    @classmethod
    def find_pos_w_rc(cls, kmer: KMerSeq, seq: DNASeq) -> NDArray[(Any,), int]:
        return np.sort(
            np.hstack(
                [
                    cls.find_pos(kmer, seq),
                    (len(seq) - 1 - cls.find_pos(kmer, rev_comp(seq))),
                ]
            )
        )

    @classmethod
    def find_pos(cls, kmer: str, seq: DNASeq) -> NDArray[(Any,), int]:
        return np.array(
            [
                m.start() + (len(kmer) - 1) // 2
                for m in re.finditer(kmer, seq, overlapped=True)
            ]
        ).astype(int)

    @classmethod
    def count_w_rc(cls, kmer: str, seqs: list[str]) -> NDArray[(Any,), int]:
        return np.array(cls.count(kmer, seqs)) + np.array(
            cls.count(kmer, rev_comp(seqs))
        )

    @classmethod
    def count(cls, kmer: str, seq: str | Iterable[str]) -> int | list[int]:
        if isinstance(seq, str):
            return len(re.findall(kmer, seq, overlapped=True))
        elif isinstance(seq, Iterable):
            return list(map(lambda c: cls.count(kmer, c), seq))
