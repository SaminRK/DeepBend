from __future__ import annotations
from pathlib import Path
import time
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from chromosome.regions import (
    Regions,
    RegionsInternal,
    PlotRegions,
    START,
    END,
    MIDDLE,
    LEN,
)
from .chromosome import Chromosome, PlotChrm
from util.reader import NucsReader
from util.util import FileSave, PathObtain
from util.constants import ONE_INDEX_START, FigSubDir

NUC_WIDTH = 147
NUC_HALF = int(NUC_WIDTH / 2)


class Nucleosomes(Regions):
    def __init__(self, chrm: Chromosome, regions: RegionsInternal = None):
        self._centers: np.ndarray = NucsReader.read(chrm.number)
        self._filter_at_least_depth(NUC_HALF, chrm.total_bp)
        super().__init__(chrm, regions)
        self.str = None
    
    def __str__(self):
        return self.str or f"nucs_w{NUC_WIDTH}"

    def _get_regions(self) -> RegionsInternal:
        return pd.DataFrame(
            {
                START: self._centers - NUC_WIDTH // 2,
                END: self._centers + NUC_WIDTH // 2,
                MIDDLE: self._centers,
            }
        )

    def _filter_at_least_depth(self, depth: int, chrm_len: int):
        """Remove center positions at each end that aren't in at least certain depth"""
        self._centers = np.array(
            list(
                filter(
                    lambda i: i > depth and i < chrm_len - depth,
                    self._centers,
                )
            )
        )

    def plot_c0_vs_dist_from_dyad_spread(self, dist=150) -> Path:
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome by
        spreading 50-bp sequence C0

        Args:
            dist: +-distance from dyad to plot (1-indexed)
        """
        spread_c0 = self.chrm.c0_spread()

        # Read C0 of -dist to +dist sequences
        c0_at_nuc: list[np.ndarray] = list(
            map(lambda c: spread_c0[c - 1 - dist : c + dist], self._centers)
        )
        assert c0_at_nuc[0].size == 2 * dist + 1
        assert c0_at_nuc[-1].size == 2 * dist + 1

        x = np.arange(dist * 2 + 1) - dist
        mean_c0 = np.array(c0_at_nuc).mean(axis=0)

        return self._plot_c0_vs_dist_from_dyad(x, mean_c0, dist, self.chrm.spread_str)

    def _plot_c0_vs_dist_from_dyad(
        self, x: np.ndarray, y: np.ndarray, dist: int, spread_str: str
    ) -> Path:
        """Underlying plotter of c0 vs dist from dyad"""
        plt.close()
        plt.clf()

        # Plot C0
        plt.plot(x, y, color="tab:blue")

        # Highlight nuc. end positions and dyad
        y_lim = plt.gca().get_ylim()
        for p in [-73, 73]:
            plt.axvline(x=p, color="tab:green", linestyle="--")
            plt.text(
                p,
                y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75,
                f"{p}bp",
                color="tab:green",
                ha="left",
                va="center",
            )

        plt.axvline(x=0, color="tab:orange", linestyle="--")
        plt.text(
            0,
            y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75,
            f"dyad",
            color="tab:orange",
            ha="left",
            va="center",
        )

        PlotChrm(self.chrm).plot_avg()
        plt.grid()

        plt.xlabel("Distance from dyad(bp)")
        plt.ylabel("C0")
        plt.title(f"C0 of +-{dist} bp from nuclesome dyad")

        return FileSave.figure(
            f"{PathObtain.figure_dir()}/nucleosome/dist_{dist}_s_{spread_str}_m_{self.chrm.predict_model_no()}_{self.chrm.id}.png"
        )

    def get_nucleosome_occupancy(self) -> np.ndarray:
        """Returns estimated nucleosome occupancy across whole chromosome

        Each dyad is extended 50 bp in both direction, resulting in a footprint
        of 101 bp for each nucleosome.
        """
        saved_data = Path(
            f"{PathObtain.data_dir()}/generated_data/nucleosome/nuc_occ_{self.chrm.id}.tsv"
        )
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep="\t")["nuc_occupancy"].to_numpy()

        centers = self._centers

        t = time.time()
        nuc_occ = np.full((self.chrm.total_bp,), fill_value=0)
        for c in centers:
            nuc_occ[c - 1 - 50 : c + 50] = 1

        print("Calculation of spread c0 balanced:", time.time() - t, "seconds.")

        # Save data
        if not saved_data.parents[0].is_dir():
            saved_data.parents[0].mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"position": np.arange(self.chrm.total_bp) + 1, "nuc_occupancy": nuc_occ}
        ).to_csv(saved_data, sep="\t", index=False)

        return nuc_occ

    # TODO: Change name to get_nuc_cover
    def get_nuc_regions(self, nuc_half: int = 73) -> np.ndarray:
        """
        Args:
            nuc_half: the region considered as within nucleosome

        Returns:
            A numpy 1D array of boolean of size chromosome total bp to
            denote nucleosome regions. An element is set to True if it
            is within +-nuc_half bp of nucleosome dyad.
        """
        self._filter_at_least_depth(nuc_half, self.chrm.total_bp)
        return self.chrm.get_cvr_mask(self._centers, nuc_half, nuc_half)

    def find_avg_nuc_linker_c0(self, nuc_half: int = 73) -> tuple[float, float]:
        """
        Find mean c0 in nuc and linker regions
        Note:
            nuc_half < 73 would mean including some nuc region with linker.
            might give less difference.
        """
        spread_c0 = self.chrm.c0_spread()
        nuc_regions = self.get_nuc_regions(nuc_half)
        nuc_avg = spread_c0[nuc_regions].mean()
        linker_avg = spread_c0[~nuc_regions].mean()
        return (nuc_avg, linker_avg)

    def dyads_between(
        self, start: int, end: int, strand: Literal[1, -1] = 1
    ) -> np.ndarray:
        """
        Get nuc dyads between start and end position (inclusive)

        Args:
            start: 1-indexed
            end: 1-indexed
            strand: Whether Watson or Crick strand. Dyads are returned
            in reverse order when strand = -1

        Returns:
            A numpy 1D array of dyad positions. (1-indexed)
        """
        dyad_arr = np.array(self._centers)
        in_between = dyad_arr[(dyad_arr >= start) & (dyad_arr <= end)]

        return in_between[::-1] if strand == -1 else in_between

    def fig_subdir(self):
        return f"{FigSubDir.NUCLEOSOMES}/{self.chrm}_{self}"

class Linkers(Regions):
    def __init__(self, chrm: Chromosome, regions: RegionsInternal = None) -> None:
        self._nucs = Nucleosomes(chrm)
        super().__init__(chrm, regions)
    
    def __str__(self):
        return f"lnks_{self._nucs}"

    def _get_regions(self) -> RegionsInternal:
        nucs = self._nucs.cover_regions()

        df = pd.DataFrame(
            {
                START: np.append([ONE_INDEX_START], nucs[END] + 1),
                END: np.append(nucs[START] - 1, self.chrm.total_bp),
            }
        )
        df[MIDDLE] = ((df[START] + df[END]) / 2).astype(int)
        return df

    def ndrs(self, len: int = 80) -> Linkers:
        return self.len_in(mn=len)


class PlotLinkers:
    def __init__(self, chrm: Chromosome) -> None:
        self._lnkrs = Linkers(chrm)

    def line_c0_indiv(self) -> None:
        for lnkr in self._lnkrs:
            PlotRegions(self._lnkrs.chrm).line_c0_indiv(lnkr)
            plt.title(f"C0 in lnkr {getattr(lnkr, START)}-{getattr(lnkr, END)}")
            FileSave.figure_in_figdir(
                f"{FigSubDir.PROMOTERS}/{self._lnkrs.chrm.id}/"
                f"{getattr(lnkr, START)}_{getattr(lnkr, END)}.png"
            )
