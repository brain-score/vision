import pytest
from pytest import approx

from brainscore.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
    MajajV4PublicBenchmark, MajajITPublicBenchmark, list_public_assemblies
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.parametrize('benchmark_ctr, visual_degrees, expected', [
    pytest.param(FreemanZiembaV1PublicBenchmark, 4, approx(.668693, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(FreemanZiembaV2PublicBenchmark, 4, approx(.596314, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(MajajV4PublicBenchmark, 8, approx(.897956, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param(MajajITPublicBenchmark, 8, approx(.816251, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self(benchmark_ctr, visual_degrees, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected


def test_list():
    assemblies = list_public_assemblies()
    assert set(assemblies) == {'dicarlo.Majaj2015.public', 'dicarlo.Majaj2015.temporal.public',
                               'movshon.FreemanZiemba2013.public', 'dicarlo.Rajalingham2018.public'}
