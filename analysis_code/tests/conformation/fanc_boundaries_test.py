import pytest

fanc = pytest.importorskip("fanc")

from conformation.fanc_boundaries import FancBoundary
from util.constants import YeastChrNumList


class TestFancBoundary:
    def test_get_boundaries(self):
        boundaries = FancBoundary()._get_all_boundaries()
        assert len(boundaries) > 0
        chrm, region = str(boundaries[0]).split(":")
        assert chrm in YeastChrNumList
        resolution_start, resolution_end = region.split("-")
        assert int(resolution_start) >= 0
        assert int(resolution_end) >= 0

    def test_get_boundaries_in(self):
        regions = FancBoundary().get_boundaries_in("XIII")
        assert len(regions) > 0
        a_region = regions[0]
        assert a_region.score > 0
        assert a_region.chromosome == "XIII"
        assert type(a_region.start) == int
        assert a_region.start > 0
        assert a_region.end > 0
        assert a_region.center > 0
