from pathlib import Path
import pytest
from pytest import approx
from brainio.stimuli import StimulusSet
from brainscore import benchmark_pool
# TODO update imports vor brainscore_vision
from brainscore.benchmarks.hebart2023 import Hebart2023Accuracy
class TestHebart2023:
    benchmark = Hebart2023Accuracy()
    assembly = benchmark._assembly

    def test_ceiling(self):
        benchmark = Hebart2023Accuracy()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.6844, abs=.0064)
