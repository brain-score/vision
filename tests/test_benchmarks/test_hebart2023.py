from pathlib import Path
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures

import sys
# Add the directory containing the file to the Python path
file_path = "/Users/linussommer/Documents/GitHub/brain-score/brainscore/benchmarks"
sys.path.append(file_path)

import hebart2023 

bm = hebart2023.Hebart2023Accuracy('dot')
assembly = bm.load_assembly()
print(assembly.attrs['stimulus_set'])
print(assembly)

"""
@pytest.mark.private_access
class TestHebart2023:
    benchmark = Hebart2023Accuracy('dot')

    # ensure the benchmark itself is there
    @pytest.mark.parametrize('benchmark', [
        'Hebart2023'
    ])
    def test_in_pool(self, benchmark):
        assert benchmark in benchmark_pool

    # Test expected ceiling
    def test_ceiling(self):
        ceiling = self.benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.# TODO, abs=.01)
    
    def start_task(self, task: BrainModel.Task):
        assert task == BrainModel.Task.odd_one_out
    

    # Test shape of BehavioralAssembly

    # Tets shape of StimulusSet


"""