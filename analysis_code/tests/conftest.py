import pytest
import pandas as pd

from chromosome.chromosome import Chromosome, C0Spread
from conformation.domains import BoundariesHE, BndParm
from chromosome.regions import Regions, START, END


@pytest.fixture
def chrm_vl():
    return Chromosome("VL")


@pytest.fixture
def chrm_vl_mean7():
    return Chromosome("VL", spread_str="mean7")


@pytest.fixture
def chrm_vl_mcvr():
    return Chromosome("VL", spread_str=C0Spread.mcvr)


@pytest.fixture
def chrm_i():
    return Chromosome("I")


@pytest.fixture
def bndrs_hirs_vl(chrm_vl_mean7: Chromosome):
    return BoundariesHE(chrm_vl_mean7, **BndParm.HIRS_WD_50)


@pytest.fixture
def rgns_simp_vl(chrm_vl_mean7):
    regions = pd.DataFrame({START: [7, 3, 9], END: [12, 4, 10]})
    return Regions(chrm_vl_mean7, regions=regions)


@pytest.fixture
def rgns_simp_vl_2(chrm_vl_mean7):
    regions = pd.DataFrame({START: [4, 6, 9], END: [6, 8, 11]})
    return Regions(chrm_vl_mean7, regions=regions)


def pytest_assertion_pass(item, lineno: int, orig: str, expl: str):
    print(f"{item} passed. Explanation: {expl}")
