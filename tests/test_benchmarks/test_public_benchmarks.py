import pytest
from pytest import approx

from brainscore.benchmarks.public_benchmarks import list_public_assemblies, \
    FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
    MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.parametrize('benchmark_ctr, visual_degrees, expected', [
    pytest.param(FreemanZiembaV1PublicBenchmark, 4, approx(.679954, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(FreemanZiembaV2PublicBenchmark, 4, approx(.577498, abs=.001),
                 marks=[pytest.mark.memory_intense]),
    pytest.param(MajajHongV4PublicBenchmark, 8, approx(.897956, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param(MajajHongITPublicBenchmark, 8, approx(.816251, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self(benchmark_ctr, visual_degrees, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected


def test_list():
    assemblies = list_public_assemblies()
    assert set(assemblies) == {'dicarlo.MajajHong2015.public', 'dicarlo.MajajHong2015.temporal.public',
                               'movshon.FreemanZiemba2013.public', 'movshon.FreemanZiemba2013.noaperture.public',
                               'dicarlo.Rajalingham2018.public', 'dicarlo.Kar2018coco_behavior.public'}
