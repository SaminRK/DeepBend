from __future__ import annotations
from util.custom_types import ChrId, YeastChrNum
from util.util import FileSave, PathObtain

# from chromosome.chromosome import Chromosome
# from models.prediction import Prediction
from util.constants import ChrIdList

import fanc
import fanc.plotting as fancplot
import pandas as pd
import numpy as np


import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class FancBoundary:
    """
    Finding boundary with fanc

    To use this class, you'll need to install `hdf5` library and `fanc` package.
    Check here - https://fan-c.readthedocs.io/en/latest/getting_started.html
    """

    def __init__(self, resolution: int = 500, window_size: int = 1000):
        self._hic_file = fanc.load(
            f"{PathObtain.hic_data_dir()}/GSE151553_A364_merged.juicer.hic@{resolution}"
        )
        self._window_size = window_size
        self._resolution = resolution
        self._insul_file = f"insulation_fanc_{self}"
        self._insulation = None
        self._insul_bed = (
            f"{PathObtain.gen_data_dir()}/hic/{self._insul_file}"
            f".insulation_{int(self._window_size / 1000)}kb.bed"
        )
        self.boundaries = self._get_all_boundaries()

    def __str__(self):
        return f"res_{self._resolution}_w_{self._window_size}"

    def plot_boundaries(
        self, chrm_num: YeastChrNum, start: int, end: int, plot_ins: bool = False
    ):
        ph = fancplot.TriangularMatrixPlot(
            self._hic_file, max_dist=50000, vmin=0, vmax=int(self._resolution / 10)
        )
        pb = fancplot.BarPlot(self.boundaries, colors="red", plot_kwargs={"alpha": 0.8})
        pl = [ph, pb]
        if plot_ins:
            self._save_ins_bed()
            pl += [fancplot.LinePlot(fanc.load(self._insul_bed))]
        f = fancplot.GenomicFigure(pl)
        if start > 0 and end > 0:
            fig, axes = f.plot(f"{chrm_num}:{int(start / 1000)}kb-{int(end / 1000)}kb")
            FileSave.figure_in_figdir(
                f"hic/{chrm_num}_{int(start / 1000)}kb_{int(end / 1000)}kb_{self}.png"
            )
        else:
            fig, axes = f.plot(f"{chrm_num}")
            FileSave.figure_in_figdir(f"hic/{chrm_num}_boundaries_{self}.png")

    def _get_insulation(self) -> fanc.InsulationScores:
        insulation_output_path = f"{PathObtain.gen_data_dir()}/hic/{self._insul_file}"
        if self._insulation is None:
            if Path(insulation_output_path).is_file():
                self._insulation = fanc.load(insulation_output_path)
            else:
                FileSave.make_parent_dirs(insulation_output_path)
                self._insulation = fanc.InsulationScores.from_hic(
                    self._hic_file,
                    [self._window_size],
                    file_name=insulation_output_path,
                )

        return self._insulation

    def _save_ins_bed(self):
        if not Path(self._insul_bed).is_file():
            self._get_insulation().to_bed(self._insul_bed, self._window_size)

    def _get_all_boundaries(self) -> fanc.architecture.domains.Boundaries:
        boundary_file_path = (
            f"{PathObtain.data_dir()}/generated_data/hic/boundaries_fanc_{self}"
        )
        if Path(boundary_file_path).is_file():
            return fanc.load(boundary_file_path)

        insulation = self._get_insulation()
        return fanc.Boundaries.from_insulation_score(
            insulation, window_size=self._window_size, file_name=boundary_file_path
        )

    def save_all_boundaries(self) -> None:
        df = pd.DataFrame(
            {
                "chromosome": list(map(lambda r: r.chromosome, self.boundaries)),
                "start": list(map(lambda r: r.start, self.boundaries)),
                "end": list(map(lambda r: r.end, self.boundaries)),
                "score": list(map(lambda r: r.score, self.boundaries)),
            }
        )
        FileSave.tsv(
            df,
            f"{PathObtain.gen_data_dir()}/boundaries/chrmall_{self}_fanc.tsv",
        )

    def get_boundaries_in(self, chrm_num: YeastChrNum) -> list[fanc.GenomicRegion]:
        return list(
            filter(lambda region: chrm_num == region.chromosome, self.boundaries)
        )

    def _get_octiles(self, regions: list[fanc.GenomicRegion] = None):
        """
        Returns:
            A numpy 1D array of size 9 denoting octiles of scores
        """
        if regions is None:
            regions = self.boundaries
        scores = np.array(list(map(lambda r: r.score, regions)))
        scores = scores[~np.isnan(scores)]
        return np.percentile(scores, np.arange(9) / 8 * 100)

    def get_one_eighth_regions_in(
        self, chrm_num: YeastChrNum
    ) -> list[list[fanc.GenomicRegion]]:
        regions_chrm = self.get_boundaries_in(chrm_num)
        octiles = self._get_octiles(regions_chrm)
        return [
            list(filter(lambda r: octiles[i] < r.score <= octiles[i + 1], regions_chrm))
            for i in range(8)
        ]

    def save_global_octiles_of_scores(self) -> None:
        octiles = self._get_octiles()
        oct_cols = [f"octile_{num}" for num in range(9)]
        FileSave.tsv(
            pd.DataFrame(octiles.reshape((1, 9)), columns=oct_cols),
            f"{PathObtain.data_dir()}/generated_data/hic/boundary_octiles.tsv",
        )


class FancBoundaryAnalysis:
    def __init__(self) -> None:
        self._boundary = FancBoundary()

    def _find_c0_at_octiles_of(self, chrm_id: ChrId, lim) -> None:
        """
        Find c0 at octiles of boundaries in each chromosome
        """
        # First, find for chr V actual
        # chrm = Chromosome(chrm_id, Prediction(30))
        chrm = None
        one_eighth_regions = self._boundary.get_one_eighth_regions_in(chrm.id)
        assert len(one_eighth_regions) == 8

        c0_spread = chrm.c0_spread()

        def _mean_at(position: int) -> float:
            """Calculate mean C0 of a region around a position"""
            return c0_spread[position - 1 - lim : position + lim].mean()

        one_eighth_centers = [
            list(map(lambda r: r.center, one_eighth_regions[i])) for i in range(8)
        ]
        one_eighth_means = [
            np.mean(
                list(map(lambda center: _mean_at(int(center)), one_eighth_centers[i]))
            )
            for i in range(8)
        ]
        cols = ["avg", "lim"] + [f"eighth_{num}" for num in range(1, 9)]
        vals = np.concatenate(([c0_spread.mean(), lim], one_eighth_means))
        FileSave.append_tsv(
            pd.DataFrame(vals.reshape((1, -1)), columns=cols),
            f"{PathObtain.data_dir()}/generated_data/hic/mean_at_boundaries.tsv",
        )

    def find_c0_at_octiles(self, lim: int = 250):
        for chrm_id in ChrIdList:
            self._find_c0_at_octiles_of(chrm_id, lim)
