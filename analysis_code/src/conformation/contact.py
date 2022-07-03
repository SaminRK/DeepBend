from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Union

import pandas as pd
import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from chromosome.chromosome import Chromosome
from util.util import PathObtain, FileSave

# TODO: Rename conformation to threedim

CorrelationType = Union[Literal["own"], Literal["adjacent"], Literal["all"]]


class Contact:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._res = 400
        # TODO: Save matrix
        self._matrix = self._generate_mat()

    def correlate_with_c0(self, type: CorrelationType) -> tuple[float, float]:
        means = self._chrm.mean_c0_at_bps(
            400 * np.arange(1, len(self._matrix)), 199, 200
        )
        intns_arr = self._intensity_arr(type)
        assert intns_arr.shape == means.shape

        pearsons = np.corrcoef(means, intns_arr)[0, 1]
        spearman = spearmanr(means, intns_arr).correlation
        return pearsons, spearman

    def plot_correlation_with_c0(self, type: CorrelationType) -> Path:
        means = self._chrm.mean_c0_at_bps(
            400 * np.arange(1, len(self._matrix)), 199, 200
        )
        intns_arr = self._intensity_arr(type)
        assert intns_arr.shape == means.shape

        plt.scatter(intns_arr, means)
        plt.xscale("log")
        plt.xlabel("Avg. contact intensity with adjacent 3 segments")
        plt.ylabel("Mean C0 of segment")
        plt.title(
            f"Correlation between contact intensity at {self._res} resolution"
            f"and avg {self._chrm.c0_type} C0 in chromosome {self._chrm.number}"
        )
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/contact/corr_{type}_{self._res}_{self._chrm}.png"
        )

    def show(self) -> Path:
        # TODO: Image from single color. Not RGB.
        img = np.zeros((self._matrix.shape[0], self._matrix.shape[1], 3))
        MIN_INTENSITY_FOR_MAX_COLOR = 100
        MAX_PIXEL_INTENSITY = 255
        img[:, :, 0] = (
            self._matrix / MIN_INTENSITY_FOR_MAX_COLOR * MAX_PIXEL_INTENSITY
        ).astype(int)
        img[img > MAX_PIXEL_INTENSITY] = MAX_PIXEL_INTENSITY

        plt.imshow(img, interpolation="nearest")
        return FileSave.figure(
            f"{PathObtain.figure_dir()}/contact/observed_vc_{self._res}_{self._chrm.number}.png"
        )

    def _intensity_arr(self, type: CorrelationType) -> NDArray[(Any,)]:
        if type == "own":
            return self._matrix.diagonal()[1:]
        elif type == "adjacent":
            return (
                self._matrix.diagonal(-1)
                + self._matrix.diagonal()[1:]
                + np.append(self._matrix.diagonal(1)[1:], 0)
            ) / 3

    def _generate_mat(self) -> NDArray[(Any, Any)]:
        """
        Contact matrix is symmetric. Contact file is a triangular matrix file.
        Three columns: row, col, intensity.

        For example, if a chromosome is of length 5200 and we take 400
        resolution, it's entries might be
        0 0   8
        0 400 4.5
        400 400 17
        ...
        4800 5200 7
        5200 5200 15
        """
        saved_contact = Path(
            f"{PathObtain.data_dir()}/generated_data/contact"
            f"/observed_vc_{self._res}_{self._chrm.number}.npy"
        )

        if saved_contact.is_file():
            return np.load(saved_contact)

        df = self._load_contact()
        df[["row", "col"]] = df[["row", "col"]] / self._res
        num_rows = num_cols = int(df[["row", "col"]].max().max()) + 1
        mat: NDArray = np.full((num_rows, num_cols), 0)

        def _fill_upper_right_half_triangle():
            for i in range(len(df)):
                elem = df.iloc[i]
                mat[int(elem.row)][int(elem.col)] = elem.intensity

        def _fill_lower_left_half_triangle(mat) -> NDArray[(Any, Any)]:
            ll = np.copy(mat.transpose())
            for i in range(num_rows):
                ll[i][i] = 0

            mat += ll
            return mat

        _fill_upper_right_half_triangle()
        mat = _fill_lower_left_half_triangle(mat)

        FileSave.npy(mat, saved_contact)
        return mat

    def _load_contact(self) -> pd.DataFrame:
        df = pd.read_table(
            f"{PathObtain.data_dir()}/input_data/contact/"
            f"observed_vc_400_{self._chrm.number}.txt",
            names=["row", "col", "intensity"],
        )
        return self._remove_too_high_intensity(df)

    MAX_INTENSITY = 1500

    def _remove_too_high_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df["intensity"] < 1500]
