import pytest
import pandas as pd

from chromosome.chromosome import Chromosome
from chromosome.regions import PlotRegions, Regions, START, END
from util.constants import FigSubDir, GDataSubDir
from util.util import FileSave


class TestRegions:
    def test_cover_regions(self, chrm_vl_mcvr, rgns_simp_vl_2: Regions):
        crgn = rgns_simp_vl_2.cover_regions()
        assert crgn[START].tolist() == [4]
        assert crgn[END].tolist() == [11]

        regions = pd.DataFrame({START: [1, 6, 7], END: [4, 8, 14]})
        rgn = Regions(chrm_vl_mcvr, regions)
        crgn = rgn.cover_regions()
        assert crgn[START].tolist() == [1, 6]
        assert crgn[END].tolist() == [4, 14]

    def test_accs_iterable(self, rgns_simp_vl: Regions):
        assert rgns_simp_vl[[True, False, True]]._regions[
            [START, END]
        ].to_numpy().tolist() == [
            [3, 4],
            [9, 10],
        ]
    
    def test_mid_contained_in(self, rgns_simp_vl: Regions, rgns_simp_vl_2: Regions):
        rgns = rgns_simp_vl.mid_contained_in(rgns_simp_vl_2)
        assert len(rgns) == 2
        assert list(rgns[START]) == [7, 9]
        assert list(rgns[END]) == [12, 10]

    def test_overlaps_with_rgns(self, rgns_simp_vl: Regions, rgns_simp_vl_2: Regions):
        orgns = rgns_simp_vl.overlaps_with_rgns(rgns_simp_vl_2, 2)
        assert len(orgns) == 2
        assert list(orgns[START]) == [7, 9]
        assert list(orgns[END]) == [12, 10]

    def test_contains_loc(self, chrm_vl_mean7: Chromosome):
        containers = pd.DataFrame({START: [3, 7, 9], END: [4, 12, 10]})
        rgns = Regions(chrm_vl_mean7, regions=containers)
        assert rgns._contains_loc([4, 11, 21, 3]).tolist() == [True, True, False]

    def test_save_regions(self, chrm_vl_mean7: Chromosome):
        rgns = Regions(chrm_vl_mean7, pd.DataFrame({START: [3], END: [10]}))
        rgns.gdata_savedir = GDataSubDir.TEST
        assert rgns.save_regions().is_file()


class TestPlotRegions:
    def test_line_c0_indiv(self, chrm_vl_mean7: Chromosome):
        rgn = pd.Series({START: 5, END: 11})
        PlotRegions(chrm_vl_mean7).line_c0_indiv(rgn)
        FileSave.figure_in_figdir(f"{FigSubDir.TEST}/region.png")
