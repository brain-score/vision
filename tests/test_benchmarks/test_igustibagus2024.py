import numpy as np
import pytest
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.domain_transfer_neural import load_domain_transfer
from . import check_standard_format, PrecomputedFeatures


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestAssembly:
    def test_IT(self):
        assembly = load_domain_transfer(average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'Igustibagus2024'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 780
        assert set(assembly['object_style'].values) == {'silhouette', 'sketch', 'cartoon', 'original', 'painting',
                                                        'line_drawing', 'outline', 'convex_hull', 'mosaic'}
        assert len(assembly['neuroid']) == 110
        assert set(assembly['animal'].values) == {'Pico', 'Oleo'}
        assert assembly['background_id'].values is not None


def test_engineering():
    benchmark = benchmark_pool['Igustibagus2024.IT_readout-accuracy']
    rng = RandomState(0)
    random_features = rng.random(size=(len(benchmark._stimuli), 300))
    features = NeuroidAssembly(random_features, coords={**{
        column: ('presentation', benchmark._stimuli[column]) for column in benchmark._stimuli
    }, **{
        'neuroid_num': ('neuroid', np.arange(random_features.shape[1])),
        'neuroid_id': ('neuroid', np.arange(random_features.shape[1]))
    }}, dims=['presentation', 'neuroid'])
    features = PrecomputedFeatures(features, visual_degrees=8)
    score = benchmark(features)
    assert score.sel(aggregation='center') == approx(0.5, abs=0.001)  # chance
    assert set(score.raw.dims) == {'domain', 'split'}
