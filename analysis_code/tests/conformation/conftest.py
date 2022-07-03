import pytest

from conformation.loops import Loops


@pytest.fixture
def loops_vl(chrm_vl):
    return Loops(chrm_vl)
