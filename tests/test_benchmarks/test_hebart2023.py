from pathlib import Path
import pytest
from pytest import approx
from brainio.stimuli import StimulusSet
from brainscore import benchmark_pool
from brainscore.benchmarks.hebart2023 import Hebart2023Accuracy
class TestHebart2023:
    benchmark = Hebart2023Accuracy()
    assembly = benchmark._assembly

    def test_in_pool(self):
        assert self.benchmark in benchmark_pool

    def test_assembly(self):
        stimulus_id = self.assembly.coords["stimulus_id"]
        triplet_id = self.assembly.coords["triplet_id"]
        assert len(stimulus_id) == len(triplet_id) == 453642

        image_1 = self.assembly.coords["image_1"]
        image_2 = self.assembly.coords["image_2"]
        image_3 = self.assembly.coords["image_3"]
        assert len(image_1) == len(image_2) == len(image_3) ==453642

    def test_stimulus_set(self):
        stimulus_set = self.assembly.attrs['stimulus_set']
        assert len(stimulus_set) == 1854
        assert isinstance(stimulus_set, StimulusSet)

    def test_ceiling(self):
        benchmark = Hebart2023Accuracy()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.6844, abs=.0064)

    # TODO
    # @pytest.mark.parametrize(['model', 'expected_score'],
    #                         [
    #                              ('alexnet', .253),
    #                              ('resnet34', .37787),
    #                              ('resnet18', .3638),
    #                         ])
