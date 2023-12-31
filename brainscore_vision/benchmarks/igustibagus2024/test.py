import numpy as np
import pytest
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_vision import load_benchmark
from brainscore_vision.benchmark_helpers import PrecomputedFeatures
from brainscore_vision.benchmark_helpers.test_helper import StandardizedTests
from brainscore_vision.benchmarks.igustibagus2024.domain_transfer_neural import load_domain_transfer


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_preprocessed_assembly():
    assembly = load_domain_transfer(average_repetitions=True)
    assert assembly.attrs['stimulus_set_identifier'] == 'Igustibagus2024'
    assert len(assembly['presentation']) == 780
    assert set(assembly['object_style'].values) == {'silhouette', 'sketch', 'cartoon', 'original', 'painting',
                                                    'line_drawing', 'outline', 'convex_hull', 'mosaic'}
    assert len(assembly['neuroid']) == 110


@pytest.mark.private_access
def test_self_regression():
    standardized_tests = StandardizedTests()
    standardized_tests.self_regression_test(benchmark='Igustibagus2024-ridge', visual_degrees=8,
                                            expected=approx(1, abs=.005))


@pytest.mark.private_access
def test_engineering():
    benchmark = load_benchmark('Igustibagus2024.IT_readout-accuracy')
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
    assert score.sel(aggregation='center') == approx(0.5, abs=0.02)  # chance
    assert set(score.raw.dims) == {'domain', 'split'}
