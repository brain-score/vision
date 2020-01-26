from candidate_models.model_commitments import brain_translated_pool
from brainscore.benchmarks import benchmark_pool
import brainscore
import numpy as np
import pytest
from pytest import approx

def test_search():
    identifier = 'vgg-16'
    model = brain_translated_pool[identifier]
    benchmark = benchmark_pool['klab.Zhang2018-ObjArray']
    assembly = benchmark._assemblies

    assert assembly.attrs['stimulus_set_name'] == 'klab.Zhang2018.search_obj_array'
    assert assembly.name == 'klab.Zhang2018search_obj_array'
    assert set(assembly.dims).issuperset({'presentation', 'fixation', 'position'})
    assert assembly.shape == (4500, 8, 2)

    score = benchmark(model)
    assert score.attrs['ceiling'].sel(aggregation='center') == approx(0.4411)
