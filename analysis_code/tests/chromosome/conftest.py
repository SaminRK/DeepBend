import pytest

from chromosome.genes import Promoters


@pytest.fixture
def prmtrs_vl(chrm_vl_mean7):
    return Promoters(chrm_vl_mean7, ustr_tss=500, dstr_tss=-1)
