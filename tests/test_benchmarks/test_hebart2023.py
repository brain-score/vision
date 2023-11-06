from pathlib import Path
import pytest
from pytest import approx
from brainio.stimuli import StimulusSet
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures

import sys
# Add the directory containing the file to the Python path
file_path = "/Users/linussommer/Documents/GitHub/brain-score/brainscore/benchmarks"
sys.path.append(file_path)

from hebart2023 import Hebart2023Accuracy

class TestHebart2023:
    benchmark = Hebart2023Accuracy()
    assembly = benchmark._assembly

    def test_in_pool(self):
        # TODO: Assertion Error
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

"""
test = TestHebart2023()
test.test_assembly()
test.test_stimulus_set()
test.test_in_pool()

    # ensure the benchmark itself is there
    

    # Test expected ceiling
    def test_ceiling(self):
        ceiling = self.benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.# TODO, abs=.01)
    
    def start_task(self, task: BrainModel.Task):
        assert task == BrainModel.Task.odd_one_out
    

    # Test shape of BehavioralAssembly

    # Tets shape of StimulusSet


"""