from __future__ import annotations
from typing import Any, TypedDict, Union
from nptyping import NDArray

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from .dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode
from util.util import rev_comp
from util.reader import SEQ_COL, SEQ_NUM_COL, C0_COL
from util.custom_types import DNASeq


class SeqTarget(TypedDict):
    all_seqs: list[DNASeq]
    rc_seqs: list[DNASeq]
    target: Union[np.ndarray, None]


class OheResult(TypedDict):
    forward: np.ndarray
    reverse: np.ndarray
    target: Union[np.ndarray, None]


class Preprocess:
    def __init__(self, df: pd.DataFrame[SEQ_NUM_COL:int, SEQ_COL:str, C0_COL:float]):
        self._df = df

    def _get_sequences_target(self) -> SeqTarget:
        all_seqs = self._df[SEQ_COL].tolist()
        rc_seqs = [rev_comp(seq) for seq in all_seqs]
        target = self._df[C0_COL].to_numpy() if C0_COL in self._df else None

        return {"all_seqs": all_seqs, "target": target, "rc_seqs": rc_seqs}

    def one_hot_encode(self) -> OheResult:
        seq_and_target = self._get_sequences_target()

        forward = [
            self.oh_enc(s)
            for s in seq_and_target["all_seqs"]
        ]
        reverse = [
            self.oh_enc(s)
            for s in seq_and_target["rc_seqs"]
        ]

        print("OHE Done")
        return {
            "forward": np.stack(forward),
            "reverse": np.stack(reverse),
            "target": seq_and_target["target"],
        }
        
    
    @classmethod 
    def oh_enc(cls, s: DNASeq) -> NDArray[(Any, 4), int]:
        def _seq_to_col_mat(seq):
            return np.array(list(seq)).reshape(-1, 1)
        
        encoder = OneHotEncoder(categories="auto")
        encoder.fit(_seq_to_col_mat("ACGT"))

        return encoder.transform(_seq_to_col_mat(s)).toarray()    

    def dinucleotide_encode(self):
        # Requires fix
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        forward_sequences = mono_to_dinucleotide(new_fasta)
        reverse_sequences = mono_to_dinucleotide(rc_fasta)

        forward = dinucleotide_one_hot_encode(forward_sequences)
        reverse = dinucleotide_one_hot_encode(reverse_sequences)

        dict = {}
        dict["readout"] = self.read_readout()
        dict["forward"] = forward
        dict["reverse"] = reverse
        return dict
