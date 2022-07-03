import pandas as pd

from models.data_preprocess import Preprocess
from util.constants import CHRVL, CHRVL_LEN, RL, RL_LEN
from util.reader import DNASequenceReader


class TestPreprocess:
    def test_get_sequences_target(self):
        df = DNASequenceReader().get_processed_data()[CHRVL]
        prep = Preprocess(df)
        seq_target = prep._get_sequences_target()
        assert set(seq_target.keys()) == set(["all_seqs", "rc_seqs", "target"])
        assert len(seq_target["all_seqs"]) == CHRVL_LEN
        assert len(seq_target["rc_seqs"]) == CHRVL_LEN
        assert len(seq_target["target"]) == CHRVL_LEN

    def test_one_hot_encode(self):
        df = pd.DataFrame({"Sequence": ["ACGGT", "CATCT"]})
        prep = Preprocess(df)
        data = prep.one_hot_encode()
        assert data["forward"].shape == (2, 5, 4)
        assert data["forward"].tolist() == [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]

        assert data["reverse"].shape == (2, 5, 4)
        assert data["reverse"].tolist() == [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        ]
