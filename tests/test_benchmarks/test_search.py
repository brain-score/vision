from brainscore.benchmarks import benchmark_pool
import brainscore
import numpy as np
import pytest
from pytest import approx
from brainscore.model_interface import BrainModel

class PrecomputedSaccades(BrainModel):
    def __init__(self):
        pass

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        self.fix = kwargs['fix'] # fixation map
        self.max_fix = kwargs['max_fix'] # maximum allowed fixation excluding the very first fixation
        self.data_len = kwargs['data_len'] # Number of stimuli
        self.current_task = task

    def look_at(self, stimuli):
        cumm_perf = np.load('precomputed_cumm_perf.npy')
        saccades = np.load('precomputed_saccades.npy')

        return cumm_perf, saccades

def test_search():
    benchmark = benchmark_pool['klab.Zhang2018-object_search']
    assembly = benchmark._assemblies

    assert assembly.attrs['stimulus_set_name'] == 'klab.Zhang2018.search_obj_array'
    assert assembly.name == 'klab.Zhang2018search_obj_array'
    assert set(assembly.dims).issuperset({'presentation', 'fixation', 'position'})
    assert assembly.shape == (4500, 8, 2)

    model = PrecomputedSaccades()
    score = benchmark(model)

    assert score.attrs['ceiling'].sel(aggregation='center') == approx(0.4411)
