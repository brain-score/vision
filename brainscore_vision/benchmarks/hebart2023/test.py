import pytest
from pytest import approx

from brainio.stimuli import StimulusSet
from brainscore_vision import benchmark_registry, load_benchmark
from brainscore_vision.benchmarks.hebart2023 import Hebart2023Accuracy

def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry

def test_ceiling():
    benchmark = Hebart2023Accuracy()
    ceiling = benchmark.ceiling
    assert ceiling.sel(aggregation='center') == approx(None, abs=None)
