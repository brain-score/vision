import pytest
from pytest import approx

from brainscore.benchmarks.public_benchmarks import list_public_assemblies, \
    FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
    MajajV4PublicBenchmark, MajajITPublicBenchmark
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.parametrize('benchmark_ctr, expected', [
    pytest.param(FreemanZiembaV1PublicBenchmark, approx(.668693, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(FreemanZiembaV2PublicBenchmark, approx(.596314, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(MajajV4PublicBenchmark, approx(.897956, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param(MajajITPublicBenchmark, approx(.816251, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self(benchmark_ctr, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    score = benchmark(PrecomputedFeatures(source)).raw
    assert score.sel(aggregation='center') == expected


def test_list():
    assemblies = list_public_assemblies()
    assert set(assemblies) == {'dicarlo.Majaj2015.public', 'dicarlo.Majaj2015.temporal.public',
                               'movshon.FreemanZiemba2013.public', 'movshon.FreemanZiemba2013.noaperture.public',
                               'dicarlo.Rajalingham2018.public'}
