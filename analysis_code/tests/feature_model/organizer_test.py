from pathlib import Path
import itertools as it

import numpy as np
import pandas as pd
import pytest

from feature_model.data_organizer import (
    DataOrganizeOptions,
    ShapeOrganizerFactory,
    DataOrganizer,
    ClassificationMaker,
    TrainTestSequenceLibraries,
    SequenceLibrary,
)
from feature_model.feat_selector import FeatureSelectorFactory
from feature_model.helsep import HSAggr
from util.reader import DNASequenceReader
from util.constants import CNL, CNL_LEN, RL, TL, TL_LEN, RL_LEN
from util.util import PathObtain


class TestDataOrganizer:
    def test_get_seq_train_test(self):
        libraries = TrainTestSequenceLibraries(
            train=[
                SequenceLibrary(name=TL, quantity=50000),
                SequenceLibrary(name=CNL, quantity=15000),
            ],
            test=[SequenceLibrary(name=RL, quantity=10000)],
        )
        options = DataOrganizeOptions(k_list=[2, 3])
        feature_factory = FeatureSelectorFactory("all")
        selector = feature_factory.make_feature_selector()
        data_organizer = DataOrganizer(libraries, None, selector, options)
        X_train, X_test, y_train, y_test = data_organizer.get_seq_train_test(
            classify=False
        )

        assert X_train.shape[0] == 65000
        assert y_train.shape[0] == 65000
        assert X_test.shape[0] == 10000
        assert y_test.shape[0] == 10000

    @pytest.mark.parametrize(
        "hsaggr",
        [HSAggr.MAX, HSAggr.SUM],
    )
    def test_get_helical_sep(self, hsaggr):
        libraries = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=CNL, quantity=CNL_LEN)],
            test=[SequenceLibrary(name=RL, quantity=RL_LEN)],
        )
        options = DataOrganizeOptions(hsaggr=hsaggr)
        organizer = DataOrganizer(libraries, None, None, options)
        hel_df_train, hel_df_test = organizer._get_helical_sep()
        assert len(hel_df_train.columns) == 3 + 120 + 16
        assert len(hel_df_test.columns) == 3 + 120 + 16

        saved_train_file = Path(
            f"{PathObtain.data_dir()}/generated_data/helical_separation"
            f"/{libraries.train[0].name}_{libraries.seq_start_pos}_{libraries.seq_end_pos}_hs_{hsaggr.value}.tsv"
        )

        saved_test_file = Path(
            f"{PathObtain.data_dir()}/generated_data/helical_separation"
            f"/{libraries.test[0].name}_{libraries.seq_start_pos}_{libraries.seq_end_pos}_hs_{hsaggr.value}.tsv"
        )

        assert saved_train_file.is_file()
        assert saved_test_file.is_file()

    def test_get_kmer_count(self):
        libs = TrainTestSequenceLibraries(
            train=[SequenceLibrary(name=TL, quantity=TL_LEN)],
            test=[SequenceLibrary(name=RL, quantity=RL_LEN)],
        )

        k_list = [2, 3]
        options = DataOrganizeOptions(k_list=k_list)

        organizer = DataOrganizer(libs, None, None, options)
        train_kmer, test_kmer = organizer._get_kmer_count()
        assert len(train_kmer.columns) == 3 + 4**2 + 4**3
        assert len(test_kmer.columns) == 3 + 4**2 + 4**3

        assert len(train_kmer) == TL_LEN
        assert len(test_kmer) == RL_LEN

        for lib, k in it.product(libs.train + libs.test, k_list):
            saved_file = Path(
                f"{PathObtain.gen_data_dir()}/kmer_count"
                f"/{lib.name}_{libs.seq_start_pos}"
                f"_{libs.seq_end_pos}_kmercount_{k}.tsv"
            )

            assert saved_file.is_file()

    def test_one_hot_encode_shape(self):
        factory = ShapeOrganizerFactory("ohe", "")
        ohe_shape_encoder = factory.make_shape_organizer(None)
        enc_arr = ohe_shape_encoder._encode_shape(np.array([[3, 7, 2], [5, 1, 4]]), 3)

        assert enc_arr.shape == (2, 3, 3)

        expected = [
            [[1, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 0, 0]],
        ]
        assert enc_arr.tolist() == expected

    def test_classify(self):
        class_maker = ClassificationMaker(np.array([0.2, 0.6, 0.2]), False)
        df = pd.DataFrame({"C0": np.array([3, 9, 13, 2, 8, 4, 11])})
        cls = class_maker.classify(df)
        assert cls["C0"].tolist() == [1, 1, 2, 0, 1, 1, 2]

    def test_get_binary_classification(self):
        class_maker = ClassificationMaker(None, None)
        df = pd.DataFrame({"C0": [1, 2, 0, 1, 1, 2]})
        df = class_maker._get_binary_classification(df)
        assert df["C0"].tolist() == [1, 0, 1]

    def test_get_balanced_classes(self):
        class_maker = ClassificationMaker(None, None)
        df = pd.DataFrame({"C0": [1, 2, 0, 1, 1, 2]})
        df, _ = class_maker.get_balanced_classes(df, df["C0"].to_numpy())
        assert set(df["C0"].tolist()) == set([0, 0, 0, 1, 1, 1, 2, 2, 2])
