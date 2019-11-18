import numpy as np
import pytest
from numpy.random.mtrand import RandomState
from pytest import approx

from brainio_base.assemblies import DataAssembly
from brainscore.benchmarks.majaj2015 import DicarloMajaj2015TemporalV4PLS, DicarloMajaj2015TemporalITPLS
from brainscore.benchmarks.kar2019 import DicarloKar2019OST
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_Kar2019():
    benchmark = DicarloKar2019OST()
    rnd = RandomState(0)
    stimuli = benchmark._assembly.stimulus_set
    source = DataAssembly(rnd.rand(len(stimuli), 5, 1), coords={
        'image_id': ('presentation', stimuli['image_id']),
        'image_label': ('presentation', stimuli['image_label']),
        'truth': ('presentation', stimuli['truth']),
        'neuroid_id': ('neuroid', list(range(5))),
        'layer': ('neuroid', ['test'] * 5),
        'time_bin_start': ('time_bin', [70]),
        'time_bin_end': ('time_bin', [170]),
    }, dims=['presentation', 'neuroid', 'time_bin'])
    source.name = 'dicarlo.Kar2019'
    score = benchmark(PrecomputedFeatures(source))
    assert np.isnan(score.raw.sel(aggregation='center'))  # not a temporal model
    assert len(score.raw.raw['split']) == 10
    assert np.isnan(score.raw.raw.values).all()
    assert score.attrs['ceiling'].sel(aggregation='center') == approx(.79)


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestMajaj2015:
    def test_V4_self(self):
        benchmark = DicarloMajaj2015TemporalV4PLS()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.temporal.V4'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.710267, abs=.00001)
        raw_values = score.attrs['raw']
        assert len(raw_values['split']) == 10
        assert len(raw_values['time_bin']) == len(source['time_bin'])

    def test_IT_self(self):
        benchmark = DicarloMajaj2015TemporalITPLS()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.temporal.IT'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.600235, abs=.00001)
        raw_values = score.attrs['raw']
        assert len(raw_values['split']) == 10
        assert len(raw_values['time_bin']) == len(source['time_bin'])
