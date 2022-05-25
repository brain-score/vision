import numpy as np
import pytest
from numpy.random.mtrand import RandomState
from pytest import approx

from brainio.assemblies import DataAssembly
from brainscore.benchmarks.kar2019 import DicarloKar2019OST
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_no_time():
    benchmark = DicarloKar2019OST()
    rnd = RandomState(0)
    stimuli = benchmark._assembly.stimulus_set
    source = DataAssembly(rnd.rand(len(stimuli), 5, 1), coords={
        'stimulus_id': ('presentation', stimuli['stimulus_id']),
        'image_label': ('presentation', stimuli['image_label']),
        'truth': ('presentation', stimuli['truth']),
        'neuroid_id': ('neuroid', list(range(5))),
        'layer': ('neuroid', ['test'] * 5),
        'time_bin_start': ('time_bin', [70]),
        'time_bin_end': ('time_bin', [170]),
    }, dims=['presentation', 'neuroid', 'time_bin'])
    source.name = __name__ + ".test_notime"
    score = benchmark(PrecomputedFeatures(source, visual_degrees=8))
    assert np.isnan(score.sel(aggregation='center'))  # not a temporal model
    assert np.isnan(score.raw.sel(aggregation='center'))  # not a temporal model
    assert score.attrs['ceiling'].sel(aggregation='center') == approx(.79)


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_random_time():
    benchmark = DicarloKar2019OST()
    rnd = RandomState(0)
    stimuli = benchmark._assembly.stimulus_set
    source = DataAssembly(rnd.rand(len(stimuli), 5, 5), coords={
        'stimulus_id': ('presentation', stimuli['stimulus_id']),
        'image_label': ('presentation', stimuli['image_label']),
        'truth': ('presentation', stimuli['truth']),
        'neuroid_id': ('neuroid', list(range(5))),
        'layer': ('neuroid', ['test'] * 5),
        'time_bin_start': ('time_bin', [70, 90, 110, 130, 150]),
        'time_bin_end': ('time_bin', [90, 110, 130, 150, 170]),
    }, dims=['presentation', 'neuroid', 'time_bin'])
    source.name = __name__ + ".test_notime"
    score = benchmark(PrecomputedFeatures(source, visual_degrees=8))
    assert np.isnan(score.sel(aggregation='center'))  # not a temporal model
    assert np.isnan(score.raw.sel(aggregation='center'))  # not a temporal model
    assert score.attrs['ceiling'].sel(aggregation='center') == approx(.79)
