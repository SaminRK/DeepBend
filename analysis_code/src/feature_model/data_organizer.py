from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Union, TypedDict
from pathlib import Path
import time
import random
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from .helsep import HSAggr, HelSep
from .occurence import Occurence
from .shape import run_dna_shape_r_wrapper
from .feat_selector import FeatureSelector
from .libstat import LibStat
from util.reader import DNASequenceReader, C0_COL, SEQ_NUM_COL, SEQ_COL
from util.custom_types import LIBRARY_NAMES
from util.util import (
    FileSave,
    PathObtain,
    cut_sequence,
    DataCache,
)
from util.constants import (
    TL,
    RL,
    CNL,
    CHRVL,
    LIBL,
    GDataSubDir,
    ONE_INDEX_START,
    SEQ_LEN,
)


@dataclass
class SequenceLibrary:
    name: TL | RL | CNL | CHRVL | LIBL
    quantity: int


TRAIN_SEQL = "train"
TEST_SEQL = "test"
TRAIN_SEQL_T = str


@dataclass
class TrainTestSequenceLibraries:
    train: list[SequenceLibrary] = field(default_factory=list)
    test: list[SequenceLibrary] = field(default_factory=list)
    seq_start_pos: int = field(default=ONE_INDEX_START)
    seq_end_pos: int = field(default=SEQ_LEN)


# The reason for using TypedDict for DataOrganizeOptions over dataclass is that
# all attributes need not be set everytime. get() function will then return None
# for options that are not set.
# TODO: You can use `dataclass` with default value set to None
class DataOrganizeOptions(TypedDict, total=False):
    """
    Attributes:
        k_list: list of unit sizes to consider when using k-mers
        range_split: Range list for classification of targets. A numpy
            array of float values between 0 and 1 that sums to 1. eg. [0.3, 0.5, 0.2]
        binary_class: Whether to perform binary classification
        balance: Whether to balance classes
        c0_scale: Scalar multiplication factor of c0
    """

    k_list: list[int]
    range_split: np.ndarray
    binary_class: bool
    balance: bool
    c0_scale: float
    hsaggr: HSAggr


class ShapeOrganizer:
    """
    Encodes DNA shape values

    Attributes:
        shape_str: Name of the shape. Valid valued: 'HelT', 'ProT', 'Roll', 'MGW', 'EP'
    """

    def __init__(self, shape_str: str, library: SequenceLibrary):
        self.shape_str = shape_str
        self._library = library

    def _encode_shape_n(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Encodes a 2D shape array into integers between 0 to (n_split - 1)

        Returns:
            A numpy 2D array of integers
        """
        saved_shape_values_file = Path(
            f"{PathObtain.data_dir()}/generated_data/{self.shape_str}_possib_values.tsv"
        )
        if saved_shape_values_file.is_file():
            with open(saved_shape_values_file, "r") as f:
                possib_shape_df = pd.read_csv(f, sep="\t")
                shape_min = min(possib_shape_df[self.shape_str])
                shape_max = max(possib_shape_df[self.shape_str])
        else:
            shape_min = shape_arr.min()
            shape_max = shape_arr.max()

        shape_range = ((shape_max - shape_min) / n_split) + 0.001
        return np.floor((shape_arr - shape_min) / shape_range)

    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        return run_dna_shape_r_wrapper(df, True)[self.shape_str]


class AlphaShapeEncoder(ShapeOrganizer):
    """
    Encodes DNA shape values into alphabetical letters
    """

    def __init__(self, shape_str: str, library: SequenceLibrary):
        super().__init__(shape_str, library)

    def _encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Encodes a 2D shape array into 1D array of strings.

        The string contains letters from first n_letters letters, ['a', ... ]

        Returns:
            A numpy 1D array of binary strings
        """
        enc_shape_arr = self._encode_shape_n(shape_arr, n_split)
        enc_shape_arr += ord("a")
        return enc_shape_arr.astype(np.uint8).view(f"S{shape_arr.shape[1]}").squeeze()

    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        pass


class OheShapeEncoder(ShapeOrganizer):
    """
    Encodes DNA shape values with one-hot encoding
    """

    def __init__(self, shape_str: str, library: SequenceLibrary):
        super().__init__(shape_str, library)

    def _encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Transform shape values to discrete numbers and perform One-hot encoding

        args:
            shape_arr (numpy.Ndarray): shape values of DNA sequences
            shape_str: a str from ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
            n_split: number of discrete values

        returns:
            a numpy 3D array containing one-hot encoded shape values of sequences
        """
        encoded_shape_arr = self._encode_shape_n(shape_arr, n_split)
        ohe = OneHotEncoder()
        ohe.fit(np.array(list(range(n_split))).reshape(-1, 1))

        arr_3d = None
        for i in range(encoded_shape_arr.shape[0]):
            seq_enc_shape_arr = (
                ohe.transform(encoded_shape_arr[i].reshape(-1, 1)).toarray().transpose()
            )
            seq_enc_shape_arr = seq_enc_shape_arr.reshape(
                1, seq_enc_shape_arr.shape[0], seq_enc_shape_arr.shape[1]
            )
            if i == 0:
                arr_3d = seq_enc_shape_arr
            else:
                arr_3d = np.concatenate((arr_3d, seq_enc_shape_arr), axis=0)

        return arr_3d

    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get shape array of DNA sequences

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with one-hot encoded shape values.
        """
        shape_arr = super().prepare_shape(df)
        print("shape_arr", shape_arr.shape)

        num_shape_encode = 12  # CHANGE HERE

        saved_shape_arr_file = Path(
            f"{PathObtain.data_dir()}/generated_data/saved_arrays/{self._library.name}_{self._library.seq_start_pos}_{self._library.seq_end_pos}_{self.shape_str}_ohe.npy"
        )
        if saved_shape_arr_file.is_file():
            # file exists
            with open(saved_shape_arr_file, "rb") as f:
                X = np.load(f)
        else:
            X = self._encode_shape(shape_arr, num_shape_encode)
            with open(saved_shape_arr_file, "wb") as f:
                np.save(f, X)

        print("X.shape", X.shape)
        print(X[random.sample(range(X.shape[0]), 5)])
        assert X.shape == (shape_arr.shape[0], num_shape_encode, shape_arr.shape[1])
        return X


class ShapeNormalizer(ShapeOrganizer):
    """
    Normalizes DNA shape values into alphabetical letters
    """

    def __init__(self, shape_str: str, library):
        super().__init__(shape_str, library)

    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get shape array of DNA sequences for training model

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with normalized shape values.
        """
        shape_arr = super().prepare_shape(df)
        print("shape_arr", shape_arr.shape)

        # Normalize
        shape_arr = (shape_arr - shape_arr.min()) / (shape_arr.max() - shape_arr.min())

        return shape_arr.reshape(shape_arr.shape[0], 1, shape_arr.shape[1])


class ShapeOrganizerFactory:
    """
    Generates shape organizer object
    """

    def __init__(self, organizer_type: str, shape_str: str):
        """
        Constructor

        Args:
            organizer_type: Valid values- ['alpha', 'ohe', 'normal'].
            shape_str: Valid values- ['HelT', 'ProT', 'Roll', 'MGW', 'EP'].
        """
        self._organizer_type = organizer_type
        self._shape_str = shape_str

    def make_shape_organizer(self, library: SequenceLibrary) -> ShapeOrganizer:
        """
        Creates shape encoder object according to encode type

        Returns:
            Created shape encoder object
        """
        if self._organizer_type == "alpha":
            return AlphaShapeEncoder(self._shape_str, library)
        elif self._organizer_type == "ohe":
            return OheShapeEncoder(self._shape_str, library)
        elif self._organizer_type == "normal":
            return ShapeNormalizer(self._shape_str, library)
        else:
            raise ValueError("Organizer type not recognized")


class ClassificationMaker:
    """Helper functions for classification"""

    def __init__(self, range_split: np.ndarray, binary_class: bool):
        """
        Creates a ClassificationMaker object.
        """
        self._range_split = range_split
        self._binary_class = binary_class

    def _classify_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Encodes a 1D numpy array into discrete values

        Args:
            arr: 1D numpy array

        Returns:
            1D arr classified into integer between 0 to self._options['range_split'].size - 1
        """
        accumulated_range = np.cumsum(self._range_split)
        assert math.isclose(accumulated_range[-1], 1.0, rel_tol=1e-4)

        # Find border values for ranges
        range_arr = np.array(sorted(arr))[
            np.floor(arr.size * accumulated_range - 1).astype(int)
        ]

        return np.array([np.searchsorted(range_arr, e) for e in arr])

    def _get_binary_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select rows with first and last class

        Args:
            df: A DataFrame in which `C0` column contains classes as denoted by integer from 0 to n-1

        Returns:
            A dataframe with selected rows
        """
        df = df.copy()
        # Select rows that contain only first and last class
        n = df[C0_COL].unique().size
        df = df.loc[((df[C0_COL] == 0) | (df[C0_COL] == n - 1))]

        # Change last class's value to 1
        df[C0_COL] = df[C0_COL].apply(lambda val: val // (n - 1))

        return df

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df[C0_COL] = self._classify_arr(df[C0_COL].to_numpy())

        if self._binary_class:
            df = self._get_binary_classification(df)

        return df

    def get_balanced_classes(
        self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray
    ) -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """
        Balance data according to classes
        """
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)

        return X_resampled, y_resampled


class DataOrganizer:
    """
    Prepares features and targets.
    """

    def __init__(
        self,
        seqlibs: TrainTestSequenceLibraries,
        shape_organizer: ShapeOrganizer,
        selector: FeatureSelector,
        options: DataOrganizeOptions = {},
    ):
        """
        Creates a DataOrganizer object.

        Args:
            libraries: TrainTestSequenceLibraries object
            shape_organizer: Shape organizer object
            selector: Feature selector
            options: Options to prepare feature and target
        """
        self._seqlibs = seqlibs
        self._shape_organizer = shape_organizer
        self._feat_selector = selector
        self._options = options
        self._cut_dfs = None
        self._class_maker = (
            ClassificationMaker(options["range_split"], options["binary_class"])
            if (
                options.get("range_split") is not None
                and options.get("binary_class") is not None
            )
            else None
        )

        if not self._options.get("c0_scale"):
            self._options["c0_scale"] = 1

        if not self._options.get("hsaggr"):
            self._options["hsaggr"] = HSAggr.MAX

    def get_seq_train_test(
        self, classify: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares features and targets from DNA sequences and C0 value.
        """
        kmer_df_train, kmer_df_test = self._get_kmer_count()
        hel_df_train, hel_df_test = self._get_helical_sep()

        df_train = pd.merge(
            kmer_df_train, hel_df_train, on=[SEQ_NUM_COL, SEQ_COL, C0_COL]
        )
        df_test = pd.merge(kmer_df_test, hel_df_test, on=[SEQ_NUM_COL, SEQ_COL, C0_COL])

        assert len(df_train) == sum(
            map(lambda slib: LibStat.len(slib.name), self._seqlibs.train)
        )
        assert len(df_test) == sum(
            map(lambda slib: LibStat.len(slib.name), self._seqlibs.test)
        )

        df_train = df_train.sample(
            n=sum(map(lambda slib: slib.quantity, self._seqlibs.train))
        )
        df_test = df_test.sample(
            n=sum(map(lambda slib: slib.quantity, self._seqlibs.test))
        )

        df_train[C0_COL] = df_train[C0_COL] * self._options["c0_scale"]
        df_test[C0_COL] = df_test[C0_COL] * self._options["c0_scale"]

        if classify:
            df_train = self._class_maker.classify(df_train)
            df_test = self._class_maker.classify(df_test)

        y_train = df_train[C0_COL].to_numpy()
        y_test = df_test[C0_COL].to_numpy()

        df_train = df_train.drop(columns=[SEQ_NUM_COL, SEQ_COL, C0_COL])
        df_test = df_test.drop(columns=[SEQ_NUM_COL, SEQ_COL, C0_COL])

        # Print sample train values
        X_train = df_train.to_numpy()
        X_test = df_test.to_numpy()
        print("X", X_train.shape)
        print("5 random rows of features")
        print(X_train[random.sample(range(X_train.shape[0]), 5)])
        print("5 random targets")
        print(y_train[random.sample(range(X_train.shape[0]), 5)])

        # Select features
        self._feat_selector.fit(X_train, y_train)
        X_train = self._feat_selector.transform(X_train)
        X_test = self._feat_selector.transform(X_test)
        print("After feature selection, X_sel", X_train.shape)
        # print('Selected features:', df_train.columns[self._feat_selector.support_])

        # Balance classes
        if classify and self._options["balance"]:
            print("Before oversampling:", sorted(Counter(y_train).items()))
            X_train, y_train = self._get_balanced_classes(X_train, y_train)
            print("After oversampling:", sorted(Counter(y_train).items()))

        # Normalize features
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

        return X_train, X_test, y_train, y_test

    def _get_kmer_count(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get k-mer count features for training and test data.

        Loads if already saved.

        Returns:
            A dictionary containing train and test libraries. Has keys
            ['train', 'test', 'train_test']
        """
        cut_dfs = self._get_cut_dfs()

        saved_kmer_paths = []

        def _get_kmer_of(library: SequenceLibrary) -> pd.DataFrame:
            df = cut_dfs[library.name]
            df_kmer = df
            for k in self._options["k_list"]:
                saved_kmer_count_file = Path(
                    f"kmer_count/{library.name}_{self._seqlibs.seq_start_pos}"
                    f"_{self._seqlibs.seq_end_pos}_kmercount_{k}.tsv"
                )
                saved_kmer_paths.append(saved_kmer_count_file)

                df_kmer = df_kmer.merge(
                    DataCache.calc_df_tsv(
                        saved_kmer_count_file,
                        lambda: Occurence().find_occurence_individual(df, [k]),
                    ),
                    on=[SEQ_NUM_COL, SEQ_COL, C0_COL],
                )

            return df_kmer

        keys = ["train", "test"]
        return tuple(
            map(
                lambda libs: pd.concat(list(map(_get_kmer_of, libs)))
                if len(libs) > 0
                else pd.DataFrame(),
                [self._seqlibs.train, self._seqlibs.test],
            )
        )

    def _get_helical_sep(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return tuple(
            map(
                lambda libs: pd.concat(list(map(self._get_helical_sep_of, libs)))
                if len(libs) > 0
                else pd.DataFrame(),
                [self._seqlibs.train, self._seqlibs.test],
            )
        )

    def _get_helical_sep_of(self, lib: SequenceLibrary) -> pd.DataFrame:
        cut_dfs = self._get_cut_dfs()
        saved_helical_sep_file = Path(
            f"{GDataSubDir.HELSEP}/{lib.name}_{self._seqlibs.seq_start_pos}"
            f"_{self._seqlibs.seq_end_pos}_hs_{self._options['hsaggr'].value}.tsv"
        )

        def _df_hel():
            t = time.time()

            helsep_df_all = None
            CHUNK_SIZE = 10000
            for idx in range(0, len(cut_dfs[lib.name]), CHUNK_SIZE):
                helsep_df = HelSep().helical_sep_of(
                    cut_dfs[lib.name][SEQ_COL][idx : idx + CHUNK_SIZE].tolist(),
                    self._options["hsaggr"],
                )
                logging.info(
                    f"[Helsep]: {min(idx + CHUNK_SIZE, len(cut_dfs[lib.name]))}"
                    f" seqs helsep calc complete"
                )
                if helsep_df_all is None:
                    helsep_df_all = helsep_df
                else:
                    helsep_df_all = pd.concat([helsep_df_all, helsep_df])

            assert len(helsep_df_all) == len(cut_dfs[lib.name])
            print(f"Helical separation count time: {(time.time() - t) / 60} min")

            return pd.merge(cut_dfs[lib.name], helsep_df_all, on=SEQ_COL, how="left")

        return DataCache.calc_df_tsv(saved_helical_sep_file, _df_hel)

    def _cut_df_train(self):
        cut_dfs = self._get_cut_dfs()
        return pd.concatenate(
            list(map(lambda slib: cut_dfs[slib.name], self._seqlibs.train))
        )

    def _cut_df_test(self):
        cut_dfs = self._get_cut_dfs()
        return pd.concatenate(
            list(map(lambda slib: cut_dfs[slib.name], self._seqlibs.test))
        )

    def _get_cut_dfs(self) -> dict[LIBRARY_NAMES, pd.DataFrame]:
        """
        Reads all sequence libraries and cut sequences from start pos to end
        pos.
        """
        if self._cut_dfs is not None:
            return self._cut_dfs

        reader = DNASequenceReader()
        library_dfs = reader.get_processed_data()

        # Cut sequences in each library
        self._cut_dfs = dict(
            map(
                lambda p: (
                    p[0],
                    cut_sequence(
                        p[1],
                        self._seqlibs.seq_start_pos,
                        self._seqlibs.seq_end_pos,
                    ),
                ),
                list(library_dfs.items()),
            )
        )
        return self._cut_dfs

    def _save_classification_data(self, df: pd.DataFrame, name: str) -> None:
        """Save classification data in a tsv file for inspection"""
        k_list_str = "".join([str(k) for k in self._options["k_list"]])
        classify_str = "_".join(
            [str(int(val * 100)) for val in self._options["range_split"]]
        )
        file_name = f'{name}_{self._seqlibs["seq_start_pos"]}_{self._seqlibs["seq_end_pos"]}_kmercount_{k_list_str}_{classify_str}'

        FileSave.tsv(
            df.sort_values(C0_COL),
            f"{PathObtain.data_dir()}/generated_data/classification/{file_name}.tsv",
        )
        df = df.drop(columns=[SEQ_NUM_COL, SEQ_COL])
        FileSave.tsv(
            df.groupby(C0_COL).mean().sort_values(C0_COL),
            f"{PathObtain.data_dir()}/generated_data/kmer_count/{file_name}_mean.tsv",
        )

    def get_shape_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares features and targets from DNA shape values and C0 value.
        """
        # df = self._get_data()()
        df = None

        # Classify
        df = self._class_maker.classify(df)
        y = df[C0_COL].to_numpy()

        X = self._shape_organizer.prepare_shape(df)
        X = X.reshape(list(X.shape) + [1])

        # Balance classes
        # X, y = get_balanced_classes(X, y)

        return train_test_split(X, y, test_size=0.1)
