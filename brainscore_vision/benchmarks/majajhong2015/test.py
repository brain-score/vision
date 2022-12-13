import pytest
from pytest import approx

from brainscore_vision.metrics.ceiling import InternalConsistency
from .benchmark import load_assembly, MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
from todotests.test_benchmarks import PrecomputedFeatures


@pytest.mark.private_access
def test_IT_ceiling():
    assembly_repetitions = load_assembly(average_repetitions=False, region='IT')
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly_repetitions)
    assert ceiling.sel(aggregation='center') == approx(.82, abs=.01)


# test public benchmarks
@pytest.mark.parametrize('benchmark_ctr, visual_degrees, expected', [
    pytest.param(MajajHongV4PublicBenchmark, 8, approx(.897956, abs=.001),
                 marks=pytest.mark.memory_intense),
    pytest.param(MajajHongITPublicBenchmark, 8, approx(.816251, abs=.001),
                 marks=pytest.mark.memory_intense),
])
def test_self(benchmark_ctr, visual_degrees, expected):
    benchmark = benchmark_ctr()
    source = benchmark._assembly.copy()
    source = {benchmark._assembly.stimulus_set.identifier: source}
    score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
    assert score.sel(aggregation='center') == expected
