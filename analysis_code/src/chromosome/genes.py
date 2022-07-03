from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nptyping import NDArray

from .chromosome import Chromosome, ChrmOperator
from .nucleosomes import Nucleosomes
from .regions import PlotRegions, Regions, RegionsInternal, START, END, MEAN_C0
from util.reader import GeneReader
from util.util import FileSave, PlotUtil, PathObtain
from util.constants import GDataSubDir, FigSubDir
from util.custom_types import PosOneIdx


STRAND = "strand"
DYADS = "dyads"


class Genes(Regions):
    "Gene is taken as transcription region including UTR."

    def __init__(self, chrm: Chromosome, regions: RegionsInternal = None):
        super().__init__(chrm, regions)

    def __str__(self):
        return "genes"

    def _get_regions(
        self,
    ) -> RegionsInternal[
        START:PosOneIdx, END:PosOneIdx, STRAND:int, DYADS : list[PosOneIdx]
    ]:
        genes = GeneReader().read_transcription_regions_of(self.chrm.number)
        return self._add_dyads(genes)

    def _add_dyads(self, genes: RegionsInternal) -> RegionsInternal:
        nucs = Nucleosomes(self.chrm)
        # TODO: Nucs need not know about strand
        genes[DYADS] = genes.apply(
            lambda gene: nucs.dyads_between(gene[START], gene[END], gene[STRAND]),
            axis=1,
        )
        return genes

    def frwrd_genes(self) -> Genes:
        return self._new(self._regions.query(f"{STRAND} == 1"))

    def rvrs_genes(self) -> Genes:
        return self._new(self._regions.query(f"{STRAND} == -1"))


class PlotGenes:
    def __init__(self, chrm: Chromosome) -> None:
        self._genes = Genes(chrm)

    def plot_mean_c0_vs_dist_from_dyad(self) -> Path:
        frwrd_p1_dyads = self._genes.frwrd_genes()[DYADS].apply(lambda dyads: dyads[0])
        frwrd_mean_c0 = self._genes.chrm.mean_c0_around_bps(frwrd_p1_dyads, 600, 400)

        rvrs_p1_dyads = self._genes.rvrs_genes()[DYADS].apply(lambda dyads: dyads[0])
        rvrs_mean_c0 = self._genes.chrm.mean_c0_around_bps(rvrs_p1_dyads, 400, 600)[
            ::-1
        ]

        mean_c0 = (
            frwrd_mean_c0 * len(frwrd_p1_dyads) + rvrs_mean_c0 * len(rvrs_p1_dyads)
        ) / (len(frwrd_p1_dyads) + len(rvrs_p1_dyads))

        PlotUtil.clearfig()
        PlotUtil.show_grid()
        plt.plot(np.arange(-600, 400 + 1), mean_c0)

        plt.xlabel("Distance from dyad (bp)")
        plt.ylabel("Mean C0")
        plt.title(
            f"{self._genes.chrm.c0_type} Mean C0 around +1 dyad"
            f" in chromosome {self._genes.chrm.number}"
        )

        return FileSave.figure(
            f"{PathObtain.figure_dir()}/genes/dist_p1_dyad_{self._genes.chrm}.png"
        )


class Promoters(Regions):
    def __init__(
        self,
        chrm: Chromosome,
        ustr_tss: int = 500,
        dstr_tss: int = -1,
        regions: RegionsInternal = None,
    ) -> None:
        self._ustr_tss = ustr_tss
        self._dstr_tss = dstr_tss
        super().__init__(chrm, regions)

    gdata_savedir = GDataSubDir.PROMOTERS

    def _get_regions(self) -> pd.DataFrame[START:int, END:int, STRAND:int]:
        genes = Genes(self.chrm)
        return pd.DataFrame(
            {
                START: pd.concat(
                    [
                        genes.frwrd_genes()[START] - self._ustr_tss,
                        genes.rvrs_genes()[END] - self._dstr_tss,
                    ],
                    ignore_index=True,
                ),
                END: pd.concat(
                    [
                        genes.frwrd_genes()[START] + self._dstr_tss,
                        genes.rvrs_genes()[END] + self._ustr_tss,
                    ],
                    ignore_index=True,
                ),
                STRAND: pd.concat(
                    [genes.frwrd_genes()[STRAND], genes.rvrs_genes()[STRAND]],
                    ignore_index=True,
                ),
            }
        )

    def __str__(self) -> str:
        return f"prmtrs_us_{self._ustr_tss}_ds_{self._dstr_tss}"

    def fig_subdir(self):
        return f"{FigSubDir.PROMOTERS}/{self.chrm}_{self}"

    def frwrd(self) -> Promoters:
        return self._new(self._regions.query(f"{STRAND} == 1"))

    def rvrs(self) -> Promoters:
        return self._new(self._regions.query(f"{STRAND} == -1"))

    def _new(self, rgns: RegionsInternal) -> Promoters:
        return Promoters(self.chrm, self._ustr_tss, self._dstr_tss, rgns)

    def mean_c0(self) -> NDArray[(Any,)]:
        return np.vstack((self.frwrd().c0(), np.flip(self.rvrs().c0(), axis=1))).mean(
            axis=0
        )
    
    @classmethod
    def val_tss_align(cls, arr: NDArray[(Any, Any), float], strand: Iterable[int]):
        assert arr.shape[0] == len(strand)
        bs = (strand == 1)
        rt = np.vstack((arr[bs], np.flip(arr[~bs], axis=1)))
        assert arr.shape == rt.shape 
        return rt

class PlotPromoters:
    def __init__(self, chrm: Chromosome) -> None:
        self._prmtrs = Promoters(chrm)

    def line_c0_indiv(self) -> None:
        for prmtr in self._prmtrs:
            PlotRegions(self._prmtrs.chrm).line_c0_indiv(prmtr)
            fr = "frw" if getattr(prmtr, STRAND) == 1 else "rvs"
            plt.title(
                f"C0 in {fr} promoter {getattr(prmtr, START)}-{getattr(prmtr, END)}"
            )
            FileSave.figure_in_figdir(
                f"{FigSubDir.PROMOTERS}/{self._prmtrs.chrm.id}/"
                f"{fr}_{getattr(prmtr, START)}_{getattr(prmtr, END)}.png"
            )

    def prob_distrib_c0(self) -> Path:
        sns.distplot(self._prmtrs[MEAN_C0], hist=False, kde=True)
        return FileSave.figure_in_figdir(
            f"genes/promoters_prob_distrib_c0_{self._prmtrs}_{self._prmtrs.chrm}.png"
        )

    def hist_c0(self) -> Path:
        PlotUtil.clearfig()
        plt.hist(self._prmtrs[MEAN_C0])
        return FileSave.figure_in_figdir(
            f"genes/promoters_hist_c0_{self._prmtrs}_{self._prmtrs.chrm}.png"
        )
